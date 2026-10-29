//! The C ABI of the spacelab core. `include/spacelab.h` is the contract this file implements and
//! `include/spacelab.hpp` is the C++ face over it; read the header first, because the four rules
//! stated there — SI and radians, `sl_status` means fallible, the library never hands out memory to
//! free, no panic escapes — are what every function below is written to satisfy.
//!
//! Rust cannot export C++. C++ has no stable ABI and its name mangling is a compiler's private
//! business, so a "C++ API" is always a C boundary with a C++ header over it. That is what this is.
//! The alternative, the `cxx` crate, would generate a real C++ binding — and would be a new
//! dependency, would need a C++ compiler inside the Rust build, and would serve C++ only, where a C
//! boundary also serves C, Julia, C#, Zig and anything else with a foreign-function interface.
//!
//! Every entry point has the same shape, and the shape is the safety argument:
//!
//! 1. Validate. Null pointers, unterminated strings, out-of-range enum tags and counts whose zero
//!    has no meaning are rejected before anything is computed.
//! 2. Compute, entirely in owned Rust values.
//! 3. Write the out-parameters, last, so a failure cannot leave a caller's struct half-filled.
//!
//! and the whole of it runs inside `guard`, because a panic unwinding across the boundary is
//! undefined behaviour rather than a crash you can debug.

// The workspace forbids `unsafe_code`; a C ABI is raw pointers by construction, so this crate opts
// out — see the note in `Cargo.toml`. Everything unsafe is confined to reading a caller's pointer
// and writing a caller's out-parameter, both immediately after a null check.
#![allow(unsafe_code)]

use core::ffi::{c_char, c_void};
use std::cell::{Cell, RefCell};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use spacelab_astro::{
    Cr3bp, lambert as astro_lambert, propagate as astro_kepler, richardson_halo_guess,
};
use spacelab_astrometry::reduce::Aberration;
use spacelab_autodiff::{DoubleDouble, Scalar};
use spacelab_constellation::{
    DraimTetrahedral, FlowerConstellation, LatticeFlower, StreetOfCoverage, Walker, WalkerPattern,
    alpha_characteristic as constellation_alpha, minimum_walker as constellation_minimum_walker,
};
use spacelab_control::attitude::{AttitudeError, ControlLaw, MrpLyapunov, QuaternionPd};
use spacelab_control::closed_loop::{AttitudeLoop, AttitudeState};
use spacelab_control::delayed::{InitialValue, breakpoints, integrate_delayed};
use spacelab_control::formation::{ConsensusNetwork, Graph, GraphError, Protocol};
use spacelab_control::latency::Latency;
use spacelab_control::margin::{
    SecondOrderLoop, SmallAngleGains, worst_principal_axis as capi_worst_axis,
};
use spacelab_control::orbit::{
    ElementFeedback, LqrError, characteristic_polynomial as control_characteristic_polynomial,
    clohessy_wiltshire_plant as capi_cw_plant, lqr as capi_lqr,
};
use spacelab_core::{Angle, Epoch, Length, Mat3, NoradId, Tt, Ut1, UtcDateTime, Vec3};
use spacelab_coverage::{
    ApparentAltitude, CoverageMetrics, Refraction, TrueAltitude, Window,
    access_windows as coverage_access_windows, constellation_fom as coverage_constellation_fom,
    revisit_map,
};
use spacelab_earth::{
    Geodetic, geodetic_to_itrf as earth_geodetic_to_itrf,
    itrf_to_geodetic as earth_itrf_to_geodetic,
};
#[cfg(feature = "onnx")]
use spacelab_earthvision::CloudMasker;
use spacelab_earthvision::atmospheric_correction::{
    AtmosphericCoupling, brdf_reflectance as ev_brdf, li_sparse_reciprocal_kernel as ev_li_sparse,
    ross_thick_kernel as ev_ross_thick,
};
use spacelab_earthvision::camera::{PinholeCamera as EvCamera, triangulate as ev_triangulate};
use spacelab_earthvision::features::{
    Corner as EvCorner, HarrisConfig, Image as EvImage, detect_corners,
};
use spacelab_earthvision::mvs::plane_sweep_mvs as ev_plane_sweep;
use spacelab_earthvision::pansharpen::{
    brovey_pan_sharpen as ev_brovey, gram_schmidt_pan_sharpen as ev_gram_schmidt,
};
use spacelab_earthvision::rpc::RationalPolynomialCamera;
use spacelab_earthvision::rpc_adjust::{
    AdjustmentModel, GroundControlPoint, estimate_adjustment, forward_adjusted,
};
use spacelab_earthvision::sfm::{Observation2d, Track, build_tracks};
use spacelab_earthvision::stereo::{SgmConfig, semi_global_match as ev_sgm};
use spacelab_earthvision::swath::{
    ground_point_at as ev_ground_point_at, horizon_look_angle as ev_horizon_look_angle,
    swath_geometry as ev_swath_geometry,
};
use spacelab_elements::design::{
    SUN_SYNCHRONOUS_NODE_RATE, critical_inclination as design_critical_inclination,
    frozen_eccentricity as design_frozen_eccentricity,
    repeat_ground_track_sma as design_repeat_ground_track_sma,
    sun_synchronous_inclination as design_sun_synchronous_inclination,
};
use spacelab_elements::{BrouwerLyddane, KeplerianElements};
use spacelab_estimation::{
    AKSIM_PAVLOV_2022, Band, BistaticGeometry, ConjunctionGeometry, CoronaSeries, CoronaTerm,
    DopplerLink, DopplerReference, DopplerSignature, GaussNewton, PlasmaSignature, Quasar,
    RampSegment, RampTable, RelativeSample, ResidualModel, Srif, Station, TdoaGeometry, Terminal,
    Topocentric, averaged_doppler as estimation_averaged_doppler, azimuth as m_azimuth,
    bistatic_range as estimation_bistatic_range,
    bistatic_range_rate as estimation_bistatic_range_rate,
    closest_approaches as estimation_closest_approaches, delta_dor as estimation_delta_dor,
    elevation as m_elevation, fdoa as estimation_fdoa,
    inter_satellite_range as estimation_inter_satellite_range,
    inter_satellite_range_rate as estimation_inter_satellite_range_rate,
    measured_frequency as estimation_measured_frequency,
    radially_separated as estimation_radially_separated, range as m_range,
    range_rate as m_range_rate, sequential_range as estimation_sequential_range,
    shapiro_delay as estimation_shapiro_delay, solar_corona_delay as estimation_solar_corona_delay,
    tdoa as estimation_tdoa, usage_violations as estimation_usage_violations,
};
use spacelab_frames::{
    CioInterpolator, Gcrf, Position, StateVector, Teme, gcrf_to_itrf as frames_gcrf_to_itrf,
    teme_to_itrf as frames_teme_to_itrf,
};
use spacelab_geomagnetism::igrf::field_ecef as igrf_field_ecef;
use spacelab_gnss::higher_order_iono::second_order_group_delay_geodetic as gnss_hoi2;
use spacelab_gnss::{
    FrequencyPair, GeometryFreeDetector, GnssSystem, PolynomialFitDetector, Reduction, SlipVerdict,
    carrier_frequency as gnss_carrier_frequency, geometry_free as gnss_geometry_free,
    graphic as gnss_graphic,
};
use spacelab_gnss::{
    ObsCode, ObsHeader, RinexObs, TimeSystem, VmfGrid, dop as gnss_dop,
    glonass_l1_frequency as gnss_glonass_l1, glonass_l2_frequency as gnss_glonass_l2,
    gmf_mapping as gnss_gmf, gpt3 as gnss_gpt3, ionospheric_mapping_factor as gnss_iono_mapping,
    klobuchar_iono_delay as gnss_klobuchar, niell_mapping as gnss_niell, parse_glonass_nav,
    parse_nav, saastamoinen_zenith_hydrostatic as gnss_saastamoinen_hydrostatic,
    saastamoinen_zenith_wet as gnss_saastamoinen_wet, vmf1_mapping as gnss_vmf1,
    vmf3_mapping as gnss_vmf3,
};
use spacelab_ionosphere::{
    DateTimeComponents, GeodeticPoint, NeQuickGalileo, path_delay as iono_path_delay,
};
use spacelab_propagation::attitude::Inertia;
use spacelab_propagation::density_calibration::{DensityCorrection, DensitySample};
use spacelab_propagation::density_data::{grid_samples, orbit_mean_samples, parse_density};
use spacelab_propagation::dop853::Dop853Dense;
use spacelab_propagation::driver::{Controls, DEFAULT_MAX_STEPS, Trajectory, integrate_dop853};
use spacelab_propagation::dsst::auxiliary::DsstAuxiliary;
use spacelab_propagation::dsst::drag::DsstAtmosphericDrag;
use spacelab_propagation::ephemeris::{AnalyticMoon, AnalyticSun, Ephemeris};
use spacelab_propagation::forces::{
    Acceleration, BodyOriented, Drag, EarthFixed, ExponentialAtmosphere, J2, Newtonian, NoWind,
    Plain, PointMass, SolarRadiationPressure, ThirdBody,
};
use spacelab_propagation::ias15::{Ias15Dense, integrate_ias15_dense};
use spacelab_propagation::lifetime::{
    DEFAULT_REENTRY_ALTITUDE, DecayAtmosphere, DecayModel, Lifetime,
    ballistic_log_sensitivity as prop_ballistic_sensitivity,
    ensemble_lifetime as prop_ensemble_lifetime, lifetime as prop_lifetime,
};
use spacelab_propagation::relativity::EARTH_ANGULAR_MOMENTUM_PER_MASS;
use spacelab_propagation::sdtm::{
    SdtmDrivers, SdtmPoint,
    coefficients::{FIELD as SDTM1_FIELD, UNCERTAINTY as SDTM1_UNCERTAINTY},
};
use spacelab_propagation::space_weather::{
    density_drivers, nrlmsise_ap_array, parse_space_weather,
};
use spacelab_propagation::symplectic::yoshida8_scalar;
use spacelab_sensors::{AgileOptical, ElevationMask, OrbitBound, SideLookingSAR};
// The highest degree the embedded gravity model carries, imported rather than written down
// here. A number copied into this file is a number that goes stale when the model is
// regenerated -- which is exactly what happened to the Python bridge, whose error message
// still says 70 for a table that has held 120 since the un-truncation work.
use spacelab_propagation::geopotential_coeffs::{MODEL_GM, MODEL_MAX_DEGREE, MODEL_RE};
use spacelab_propagation::{
    DeSitter, Dop853, DsstMeanPropagator, DsstSolarRadiationPressure, DsstTesseral, DsstThirdBody,
    DsstZonal, GeoPotential, LenseThirring, MeanPole, Nrlmsise00Atmosphere, Nrlmsise00Flags,
    Nrlmsise00Input, Schwarzschild, SolidEarthTide, Tolerance,
    conventional_zonal_coefficients as tvg_conventional, gtd7d, ocean_pole_tide_corrections,
    pole_tide_corrections, secular_zonal_correction as tvg_secular, sentman_panel,
    speed_ratio as sentman_speed_ratio, step2_corrections,
};
use spacelab_sgp4::tle::{TleBuilder, TleModel, seed_elements};
use spacelab_sgp4::{Convention, Satellite};
use spacelab_signal::csa::{CsaParams, chirp_scaling_focus as sig_csa_focus};
use spacelab_signal::fft::Complex as SigComplex;
use spacelab_signal::insar::{
    coherence as sig_coherence, interferogram as sig_interferogram, residues as sig_residues,
    unwrap_phase_1d as sig_unwrap_1d, unwrap_phase_2d as sig_unwrap_2d,
    unwrap_phase_2d_branch_cut as sig_unwrap_branch_cut, unwrap_phase_2d_lstsq as sig_unwrap_lstsq,
    wrap_phase as sig_wrap_phase,
};
use spacelab_signal::polsar::{
    cloude_pottier as sig_cloude_pottier, coherency_matrix as sig_coherency,
    pauli_powers as sig_pauli_powers,
};
use spacelab_signal::psinsar::{
    amplitude_dispersion as sig_amplitude_dispersion, sbas_invert as sig_sbas_invert,
    select_persistent_scatterers as sig_select_ps,
};
use spacelab_signal::sar::apply_chirp_scaling as sig_apply_chirp_scaling;
use spacelab_signal::sar::{
    RdaParams, azimuth_fm_rate as sig_azimuth_fm_rate, lfm_chirp as sig_lfm_chirp,
    matched_filter as sig_matched_filter, peak_sidelobe_ratio_db as sig_pslr,
    range_migration as sig_range_migration, resolution_3db_samples as sig_res3db,
    src_fm_rate as sig_src_fm_rate,
};
use spacelab_signal::tomosar::{
    Sl1mmerConfig, bartlett_spectrum as sig_bartlett, capon_spectrum as sig_capon,
    sl1mmer as sig_sl1mmer, sl1mmer_spectrum as sig_sl1mmer_spectrum,
};
use spacelab_vision::detect::{
    BBox, decode_box as dv_decode_box, generate_anchors as dv_generate_anchors, nms as dv_nms,
};

/// The ABI revision. Must equal `SL_ABI_VERSION` in `spacelab.h`; a test checks that it does.
const ABI_VERSION: u32 = 2;

/// The longest NUL-terminated string this library will scan for a terminator. Past it the input is
/// rejected rather than read on: there is no way to learn a C string's buffer length, so a bound is
/// the only defence against a caller who forgot the terminator, and a two-line element set is 69
/// bytes a line.
const MAX_C_STRING: usize = 4096;

// The status codes, mirroring the `SL_*` macros in the header one for one.

/// The call succeeded.
pub const SL_OK: i32 = 0;
/// A pointer was null, a string was unterminated or not UTF-8, or a tag was out of range.
pub const SL_ERR_INVALID_ARGUMENT: i32 = 1;
/// Well-formed inputs whose answer does not exist.
pub const SL_ERR_DOMAIN: i32 = 2;
/// Input text did not parse.
pub const SL_ERR_PARSE: i32 = 3;
/// The computation started and failed.
pub const SL_ERR_COMPUTE: i32 = 4;
/// The caller's buffer is shorter than the result; `*out_len` says how long it must be.
pub const SL_ERR_BUFFER_TOO_SMALL: i32 = 5;
/// A panic was caught at the boundary.
pub const SL_ERR_PANIC: i32 = 6;
/// The entry point exists in this ABI but the library was built without the feature behind it.
pub const SL_ERR_UNSUPPORTED: i32 = 7;

/// `sl_precision`: ordinary `f64`.
const PRECISION_DOUBLE: i32 = 0;
/// `sl_precision`: double-double, ~32 significant digits.
const PRECISION_DOUBLE_DOUBLE: i32 = 1;

/// `sl_walker_pattern`: planes spread over a full turn of right ascension.
const WALKER_DELTA: i32 = 0;
/// `sl_walker_pattern`: planes spread over half a turn.
const WALKER_STAR: i32 = 1;

// ── Types ─────────────────────────────────────────────────────────────────────────────────────

/// `sl_vec3`: a Cartesian triple, metres or metres per second.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlVec3 {
    /// First component.
    pub x: f64,
    /// Second component.
    pub y: f64,
    /// Third component.
    pub z: f64,
}

/// `sl_state`: a position and velocity in one frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlState {
    /// Position, metres.
    pub position: SlVec3,
    /// Velocity, metres per second.
    pub velocity: SlVec3,
}

/// `sl_geodetic`: WGS-84 latitude, longitude and height above the ellipsoid.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlGeodetic {
    /// Geodetic latitude, radians, positive north.
    pub latitude_rad: f64,
    /// Longitude, radians, positive east.
    pub longitude_rad: f64,
    /// Height above the ellipsoid, metres.
    pub height_m: f64,
}

/// `sl_latlon`: a point on the ellipsoid, where height is not part of the answer.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlLatLon {
    /// Geodetic latitude, radians.
    pub latitude_rad: f64,
    /// Longitude, radians.
    pub longitude_rad: f64,
}

/// `sl_walker_params`: the Walker `i: T/P/F` design.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlWalkerParams {
    /// `SL_WALKER_DELTA` or `SL_WALKER_STAR`.
    pub pattern: i32,
    /// Inclination, radians.
    pub inclination_rad: f64,
    /// Number of orbital planes.
    pub planes: u32,
    /// Satellites in each plane.
    pub per_plane: u32,
    /// The inter-plane phasing parameter `F`.
    pub phasing: u32,
    /// Semi-major axis, metres.
    pub semi_major_axis_m: f64,
    /// Gravitational parameter, m³/s².
    pub mu: f64,
}

/// `sl_soc_params`: the Street-of-Coverage design.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlSocParams {
    /// Inclination, radians.
    pub inclination_rad: f64,
    /// Altitude of the circular orbit, metres.
    pub altitude_m: f64,
    /// The elevation the coverage must clear, radians.
    pub min_elevation_rad: f64,
    /// Earth radius, metres.
    pub earth_radius_m: f64,
    /// Gravitational parameter, m³/s².
    pub mu: f64,
    /// Satellites per plane, or zero for the fewest that still tile the street.
    pub per_plane: u32,
}

/// `sl_soc_geometry`: the solved Street-of-Coverage geometry.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlSocGeometry {
    /// Number of planes.
    pub planes: u32,
    /// Satellites per plane.
    pub per_plane: u32,
    /// Half-angle of one satellite's footprint, radians.
    pub footprint_half_angle_rad: f64,
    /// Half-width of the street a plane sweeps, radians.
    pub street_half_width_rad: f64,
    /// Right-ascension step between co-rotating planes, radians.
    pub raan_step_co_rad: f64,
    /// Right-ascension step across the counter-rotating seam, radians.
    pub raan_step_counter_rad: f64,
    /// The step actually used, radians.
    pub raan_step_rad: f64,
    /// Total right-ascension span the planes cover, radians.
    pub raan_span_rad: f64,
    /// Phase offset between adjacent planes, radians.
    pub inter_plane_phase_rad: f64,
}

/// `sl_sgp4`: a parsed element set. Opaque to C; the layout is not part of the ABI.
pub struct SlSgp4 {
    satellite: Satellite,
}

// `spacelab.h` promises that a handle may be shared across threads and propagated concurrently.
// Across an FFI boundary the type is erased, so nothing on the C side can be told otherwise -- a
// `Cell` added to `Satellite` later as a cache would turn that sentence into a data race with no
// compiler complaint anywhere. This is the complaint: it stops compiling the day the promise stops
// being true, which is the only moment anyone would want to hear about it.
const _: () = {
    const fn assert_shareable<T: Send + Sync>() {}
    assert_shareable::<SlSgp4>();
};

// ── Errors ────────────────────────────────────────────────────────────────────────────────────

thread_local! {
    /// The last failure on this thread. Thread-local rather than global so that a failure on one
    /// thread is never read by another — a global here would be a data race dressed as a
    /// convenience, and the whole API is documented as callable concurrently.
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Record `message` as this thread's last failure and hand back `status`, so that every failing
/// path is one `return fail(...)` and cannot set one without the other.
fn fail(status: i32, message: impl Into<String>) -> i32 {
    let message = message.into();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message);
    status
}

/// Run `body` with panics contained. A panic crossing into C++ is undefined behaviour, not a crash
/// you can debug, so the boundary is where it stops.
fn guard(body: impl FnOnce() -> i32) -> i32 {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(status) => status,
        Err(payload) => fail(
            SL_ERR_PANIC,
            format!("panic at the C boundary: {}", describe(&*payload)),
        ),
    }
}

/// The same for a total function, whose only way to report a contained panic is the NaN it already
/// returns for a NaN input.
fn guard_f64(body: impl FnOnce() -> f64) -> f64 {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            let _ = fail(
                SL_ERR_PANIC,
                format!("panic at the C boundary: {}", describe(&*payload)),
            );
            f64::NAN
        }
    }
}

/// The human-readable half of a panic payload, which is a `&str` or a `String` for every panic the
/// standard macros raise and opaque for the rest.
fn describe(payload: &(dyn core::any::Any + Send)) -> &str {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.as_str()
    } else {
        "(payload is not a string)"
    }
}

/// Copy `text` into a caller's `buf` as a NUL-terminated C string, returning the bytes the whole
/// string needs including the terminator — so `copy_to_c_buffer(text, null, 0)` sizes it.
///
/// Truncation stops on a UTF-8 boundary. Cutting mid-sequence would leave the caller holding bytes
/// that are a valid C string and an invalid `std::string`, which is the kind of difference that is
/// found in production rather than in a test.
fn copy_to_c_buffer(text: &str, buf: *mut c_char, cap: usize) -> usize {
    let needed = text.len() + 1;
    if buf.is_null() || cap == 0 {
        return needed;
    }
    let mut keep = core::cmp::min(text.len(), cap - 1);
    while keep > 0 && !text.is_char_boundary(keep) {
        keep -= 1;
    }
    // SAFETY: the caller states `buf` is writable for `cap` bytes, and `keep < cap`.
    unsafe {
        core::ptr::copy_nonoverlapping(text.as_ptr(), buf.cast::<u8>(), keep);
        *buf.add(keep) = 0;
    }
    needed
}

/// Borrow a NUL-terminated C string, scanning no further than [`MAX_C_STRING`] for the terminator.
///
/// # Safety
/// `ptr` is either null or the start of a NUL-terminated buffer. Nothing can verify the second
/// half — C strings carry no length — so the bound is the defence: an unterminated buffer costs at
/// most `MAX_C_STRING` bytes of over-read instead of running to the end of the address space.
unsafe fn borrow_c_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    for len in 0..MAX_C_STRING {
        // SAFETY: every byte up to and including this one has been non-zero, so the terminator (and
        // therefore the buffer) extends at least this far.
        if unsafe { *ptr.add(len) } == 0 {
            // SAFETY: `len` bytes precede a terminator inside the caller's buffer.
            let bytes = unsafe { core::slice::from_raw_parts(ptr.cast::<u8>(), len) };
            return core::str::from_utf8(bytes).ok();
        }
    }
    None
}

fn to_vec3(v: SlVec3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

fn from_vec3(v: Vec3) -> SlVec3 {
    SlVec3 {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}

fn from_state(position: Vec3, velocity: Vec3) -> SlState {
    SlState {
        position: from_vec3(position),
        velocity: from_vec3(velocity),
    }
}

/// The six-element form the measurement models take.
fn to_six(s: SlState) -> [f64; 6] {
    [
        s.position.x,
        s.position.y,
        s.position.z,
        s.velocity.x,
        s.velocity.y,
        s.velocity.z,
    ]
}

fn to_three(v: SlVec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

// ── Preconditions ─────────────────────────────────────────────────────────────────────────────
//
// Every predicate an entry point applies to a value lives in a safe function, here or beside the
// entry point that needs it. That is not a style preference: `cargo mutants` 27.1.0 generates
// nothing for the body of an `unsafe fn`, and 146 of this crate's 164 exports are `unsafe`, so a
// condition written inline in one of them cannot be measured — deleting it and asserting on it look
// the same to the seal. Written here, each predicate is mutated, and a test has to kill it (#260).
//
// What stays inline is what cannot move: null tests, `borrow_sized_*`, and the arms that report a
// *computation* failing rather than an input being wrong.

/// `value` must be finite. The message names the input and the number it arrived as, because
/// "invalid argument" without the value is a support ticket rather than a diagnosis.
fn check_finite(who: &str, name: &str, value: f64) -> Result<(), i32> {
    if !value.is_finite() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: {name} is {value}, and must be finite"),
        ));
    }
    Ok(())
}

/// `value` must be finite and strictly positive.
///
/// The status is the caller's rather than fixed here: this crate answers `SL_ERR_DOMAIN` where a
/// well-formed number is one the model has nothing to say about, and `SL_ERR_INVALID_ARGUMENT`
/// where it is not a well-formed input at all. Which of the two a given entry point returns is
/// observable to a C caller, so it is preserved per entry point rather than unified here.
fn check_positive(who: &str, name: &str, value: f64, status: i32) -> Result<(), i32> {
    if !(value.is_finite() && value > 0.0) {
        return Err(fail(
            status,
            format!("{who}: {name} is {value}, and must be positive"),
        ));
    }
    Ok(())
}

/// Every element of `values` must be finite, and the refusal names the first one that is not.
fn check_all_finite(who: &str, name: &str, values: &[f64]) -> Result<(), i32> {
    if let Some(i) = values.iter().position(|v| !v.is_finite()) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: {name}[{i}] is {}, and must be finite", values[i]),
        ));
    }
    Ok(())
}

/// `value` must be finite and not negative. Zero is admitted, and that is the difference from
/// `check_positive`: for a ridge weight, a tolerance or a floor, zero is a meaningful request.
fn check_nonnegative(who: &str, name: &str, value: f64) -> Result<(), i32> {
    if !(value.is_finite() && value >= 0.0) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: {name} is {value}, and must be finite and non-negative"),
        ));
    }
    Ok(())
}

/// A count with an algorithmic floor above one — interpolation needs two rows, triangulation two
/// views — rather than merely non-zero.
fn check_at_least(who: &str, name: &str, n: usize, minimum: usize) -> Result<(), i32> {
    if n < minimum {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: {name} is {n}, and must be at least {minimum}"),
        ));
    }
    Ok(())
}

/// A length the ABI fixes exactly, because the array behind the pointer has a compile-time size and
/// the count is the caller's assertion about it rather than a free choice.
fn check_count(who: &str, name: &str, n: usize, expected: usize) -> Result<(), i32> {
    if n != expected {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: {name} is {n}, and must be exactly {expected}"),
        ));
    }
    Ok(())
}

/// A count that a loop, an allocation or a division is about to depend on must not be zero.
///
/// Generic over the integer because the ABI's counts are `size_t`, `uint32_t` and `uint64_t`
/// according to what each one counts, and a cast at the call site would be a place to get it wrong.
/// `Default` is zero for every integer type.
fn check_nonzero<T: Default + PartialEq>(who: &str, name: &str, n: T) -> Result<(), i32> {
    if n == T::default() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: {name} is zero"),
        ));
    }
    Ok(())
}

// ── ABI version ───────────────────────────────────────────────────────────────────────────────

/// The ABI revision this library was built with.
#[unsafe(no_mangle)]
pub extern "C" fn sl_abi_version() -> u32 {
    ABI_VERSION
}

/// Write the library version as a C string; see the header for the sizing protocol.
///
/// # Safety
/// `buf` is null, or writable for `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_version_string(buf: *mut c_char, cap: usize) -> usize {
    copy_to_c_buffer(env!("CARGO_PKG_VERSION"), buf, cap)
}

/// Write this thread's last failure message as a C string; see the header for the sizing protocol.
///
/// # Safety
/// `buf` is null, or writable for `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_last_error_message(buf: *mut c_char, cap: usize) -> usize {
    LAST_ERROR.with(|slot| copy_to_c_buffer(&slot.borrow(), buf, cap))
}

// ── Time and frames ───────────────────────────────────────────────────────────────────────────

/// TEME to ITRF. See the header.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `out` is writable for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_teme_to_itrf(
    r#in: *const SlState,
    ut1_ns: i64,
    tt_ns: i64,
    xp_rad: f64,
    yp_rad: f64,
    lod_seconds: f64,
    out: *mut SlState,
) -> i32 {
    guard(|| {
        if r#in.is_null() || out.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_teme_to_itrf: null state pointer",
            );
        }
        // SAFETY: checked non-null just above; the caller states it is initialised.
        let state = unsafe { *r#in };
        let transform = frames_teme_to_itrf(
            Epoch::<Ut1>::from_ns_since_j2000(i128::from(ut1_ns)),
            Epoch::<Tt>::from_ns_since_j2000(i128::from(tt_ns)),
            Angle::from_radians(xp_rad),
            Angle::from_radians(yp_rad),
            lod_seconds,
        );
        let sv = transform.state(StateVector::<Teme>::new(
            to_vec3(state.position),
            to_vec3(state.velocity),
        ));
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe { *out = from_state(sv.position.vector, sv.velocity.vector) };
        SL_OK
    })
}

/// GCRF to ITRF by the IAU 2006/2000A CIO chain. See the header.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `out` is writable for one.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)] // a state, an epoch on two scales, and the five EOP quantities
pub unsafe extern "C" fn sl_gcrf_to_itrf(
    r#in: *const SlState,
    ut1_ns: i64,
    tt_ns: i64,
    xp_rad: f64,
    yp_rad: f64,
    dx_rad: f64,
    dy_rad: f64,
    lod_seconds: f64,
    out: *mut SlState,
) -> i32 {
    guard(|| {
        if r#in.is_null() || out.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_gcrf_to_itrf: null state pointer",
            );
        }
        // SAFETY: checked non-null just above; the caller states it is initialised.
        let state = unsafe { *r#in };
        let transform = frames_gcrf_to_itrf(
            Epoch::<Ut1>::from_ns_since_j2000(i128::from(ut1_ns)),
            Epoch::<Tt>::from_ns_since_j2000(i128::from(tt_ns)),
            Angle::from_radians(xp_rad),
            Angle::from_radians(yp_rad),
            Angle::from_radians(dx_rad),
            Angle::from_radians(dy_rad),
            lod_seconds,
        );
        let sv = transform.state(StateVector::<Gcrf>::new(
            to_vec3(state.position),
            to_vec3(state.velocity),
        ));
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe { *out = from_state(sv.position.vector, sv.velocity.vector) };
        SL_OK
    })
}

/// ITRF Cartesian to WGS-84 geodetic. See the header.
///
/// # Safety
/// `r_itrf` points at one initialised `sl_vec3`; `out` is writable for one `sl_geodetic`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_itrf_to_geodetic(r_itrf: *const SlVec3, out: *mut SlGeodetic) -> i32 {
    guard(|| {
        if r_itrf.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_itrf_to_geodetic: null pointer");
        }
        // SAFETY: checked non-null just above.
        let r = unsafe { *r_itrf };
        match earth_itrf_to_geodetic(Position::new(to_vec3(r))) {
            Ok(g) => {
                // SAFETY: checked non-null; written only now that the answer exists.
                unsafe {
                    *out = SlGeodetic {
                        latitude_rad: g.latitude.radians(),
                        longitude_rad: g.longitude.radians(),
                        height_m: g.height.metres(),
                    }
                };
                SL_OK
            }
            Err(e) => fail(SL_ERR_COMPUTE, format!("sl_itrf_to_geodetic: {e}")),
        }
    })
}

/// WGS-84 geodetic to an ITRF Cartesian position. See the header.
///
/// # Safety
/// `g` points at one initialised `sl_geodetic`; `out` is writable for one `sl_vec3`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_geodetic_to_itrf(g: *const SlGeodetic, out: *mut SlVec3) -> i32 {
    guard(|| {
        if g.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_geodetic_to_itrf: null pointer");
        }
        // SAFETY: checked non-null just above.
        let g = unsafe { *g };
        let p = earth_geodetic_to_itrf(Geodetic {
            latitude: Angle::from_radians(g.latitude_rad),
            longitude: Angle::from_radians(g.longitude_rad),
            height: Length::from_metres(g.height_m),
        });
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe { *out = from_vec3(p.vector) };
        SL_OK
    })
}

// ── SGP4 ──────────────────────────────────────────────────────────────────────────────────────

/// Parse a two-line element set into a handle. See the header.
///
/// # Safety
/// `line1` and `line2` are NUL-terminated C strings; `out` is writable for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sgp4_open(
    line1: *const c_char,
    line2: *const c_char,
    out: *mut *mut SlSgp4,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_sgp4_open: null out pointer");
        }
        // SAFETY: the caller states these are NUL-terminated or null; `borrow_c_str` handles both.
        let (Some(l1), Some(l2)) = (unsafe { borrow_c_str(line1) }, unsafe {
            borrow_c_str(line2)
        }) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_sgp4_open: a TLE line was null, unterminated within 4096 bytes, or not UTF-8",
            );
        };
        match Satellite::from_tle(l1, l2) {
            Ok(satellite) => {
                let handle = Box::into_raw(Box::new(SlSgp4 { satellite }));
                // SAFETY: checked non-null; written only now that the handle exists.
                unsafe { *out = handle };
                SL_OK
            }
            Err(e) => fail(SL_ERR_PARSE, format!("sl_sgp4_open: invalid TLE: {e:?}")),
        }
    })
}

/// Release a handle from [`sl_sgp4_open`].
///
/// # Safety
/// `sat` is null, or a handle from `sl_sgp4_open` that has not already been closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sgp4_close(sat: *mut SlSgp4) {
    if sat.is_null() {
        return;
    }
    // SAFETY: the caller states this came from `Box::into_raw` in `sl_sgp4_open` and is unclosed.
    drop(unsafe { Box::from_raw(sat) });
}

/// Propagate a handle to one offset in minutes. See the header.
///
/// # Safety
/// `sat` is a live handle; `out` is writable for one `sl_state`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sgp4_propagate_at(
    sat: *const SlSgp4,
    minutes: f64,
    out: *mut SlState,
) -> i32 {
    guard(|| {
        if sat.is_null() || out.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_sgp4_propagate_at: null pointer",
            );
        }
        // SAFETY: checked non-null; the caller states the handle is live.
        let sat = unsafe { &*sat };
        match sat.satellite.propagate_minutes(minutes) {
            Ok(sv) => {
                // SAFETY: checked non-null; written only now that the answer exists.
                unsafe { *out = from_state(sv.position.vector, sv.velocity.vector) };
                SL_OK
            }
            Err(e) => fail(
                SL_ERR_COMPUTE,
                format!("sl_sgp4_propagate_at: SGP4 failed at {minutes} min: {e:?}"),
            ),
        }
    })
}

/// Propagate a handle to `n` offsets at once. See the header.
///
/// # Safety
/// `sat` is a live handle; `minutes` is readable for `n` doubles; `out` is writable for `n`
/// `sl_state` and does not overlap `minutes`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sgp4_propagate_batch(
    sat: *const SlSgp4,
    minutes: *const f64,
    n: usize,
    out: *mut SlState,
) -> i32 {
    guard(|| {
        if sat.is_null() || minutes.is_null() || out.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_sgp4_propagate_batch: null pointer",
            );
        }
        // SAFETY: checked non-null; the caller states the handle is live and the slice is `n` long.
        let sat = unsafe { &*sat };
        let offsets = unsafe { core::slice::from_raw_parts(minutes, n) };

        // Collected before anything is written, so that the documented "nothing was written on
        // failure" holds for the batch as a whole and not merely up to the offset that failed.
        let mut states = Vec::with_capacity(n);
        for &m in offsets {
            match sat.satellite.propagate_minutes(m) {
                Ok(sv) => states.push(from_state(sv.position.vector, sv.velocity.vector)),
                Err(e) => {
                    return fail(
                        SL_ERR_COMPUTE,
                        format!("sl_sgp4_propagate_batch: SGP4 failed at {m} min: {e:?}"),
                    );
                }
            }
        }
        // SAFETY: `out` is writable for `n` and `states.len() == n`; the buffers do not overlap.
        unsafe { core::ptr::copy_nonoverlapping(states.as_ptr(), out, n) };
        SL_OK
    })
}

/// Parse and propagate in one call. See the header.
///
/// # Safety
/// `line1` and `line2` are NUL-terminated C strings; `out` is writable for one `sl_state`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sgp4_propagate(
    line1: *const c_char,
    line2: *const c_char,
    minutes: f64,
    out: *mut SlState,
) -> i32 {
    let mut handle: *mut SlSgp4 = core::ptr::null_mut();
    // SAFETY: forwarding the caller's own contract, with a local `out` this function owns.
    let status = unsafe { sl_sgp4_open(line1, line2, &raw mut handle) };
    if status != SL_OK {
        return status;
    }
    // SAFETY: `handle` is live because `sl_sgp4_open` returned `SL_OK`.
    let status = unsafe { sl_sgp4_propagate_at(handle, minutes, out) };
    // SAFETY: the same handle, closed exactly once, on both the success and the failure path.
    unsafe { sl_sgp4_close(handle) };
    status
}

// ── Two-body motion ───────────────────────────────────────────────────────────────────────────

/// Universal-variable Kepler propagation. See the header.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `out` is writable for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_kepler_propagate(
    r#in: *const SlState,
    mu: f64,
    dt: f64,
    out: *mut SlState,
) -> i32 {
    guard(|| {
        if r#in.is_null() || out.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_kepler_propagate: null state pointer",
            );
        }
        // SAFETY: checked non-null just above.
        let s = unsafe { *r#in };
        let (r, v) = astro_kepler(to_vec3(s.position), to_vec3(s.velocity), mu, dt);
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe { *out = from_state(r, v) };
        SL_OK
    })
}

/// Which arithmetic an `sl_precision` tag names.
enum Precision {
    Double,
    DoubleDouble,
}

/// The precision tag as the choice it names, or the refusal that says it names none.
///
/// Decoding the tag in a safe helper rather than in the `match` that dispatches on it is what puts
/// the check inside the mutation seal: `cargo mutants` generates nothing for the body of an
/// `unsafe fn`, so a catch-all arm written in an entry point cannot be measured. The same reason
/// governs every `check_*` below.
fn precision_of(who: &str, tag: i32) -> Result<Precision, i32> {
    match tag {
        PRECISION_DOUBLE => Ok(Precision::Double),
        PRECISION_DOUBLE_DOUBLE => Ok(Precision::DoubleDouble),
        other => Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: precision tag {other} is not an sl_precision"),
        )),
    }
}

/// A step count as a `usize`, or the refusal that says this platform cannot hold it.
fn check_steps(who: &str, steps: u64) -> Result<usize, i32> {
    usize::try_from(steps).map_err(|_| {
        fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: {steps} steps does not fit this platform's size_t"),
        )
    })
}

/// Fixed-step symplectic two-body propagation, in either precision. See the header.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `out` is writable for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_propagate_two_body(
    r#in: *const SlState,
    mu: f64,
    dt: f64,
    steps: u64,
    precision: i32,
    out: *mut SlState,
) -> i32 {
    guard(|| {
        if r#in.is_null() || out.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_propagate_two_body: null state pointer",
            );
        }
        let steps = match check_steps("sl_propagate_two_body", steps) {
            Ok(steps) => steps,
            Err(status) => return status,
        };
        let precision = match precision_of("sl_propagate_two_body", precision) {
            Ok(precision) => precision,
            Err(status) => return status,
        };
        // SAFETY: checked non-null just above.
        let s = unsafe { *r#in };
        let r0 = to_three(s.position);
        let v0 = to_three(s.velocity);

        let (r, v) = match precision {
            Precision::Double => {
                let accel = |x: &[f64; 3]| {
                    let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                    let k = -mu / (r2 * r2.sqrt());
                    [k * x[0], k * x[1], k * x[2]]
                };
                let (mut x, mut v) = (r0, v0);
                yoshida8_scalar(&accel, &mut x, &mut v, dt, steps);
                (x, v)
            }
            Precision::DoubleDouble => {
                let neg_mu = DoubleDouble::from_f64(-mu);
                let accel = |x: &[DoubleDouble; 3]| {
                    let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                    let k = neg_mu / (r2 * r2.sqrt());
                    [k * x[0], k * x[1], k * x[2]]
                };
                let mut x = r0.map(DoubleDouble::from_f64);
                let mut v = v0.map(DoubleDouble::from_f64);
                yoshida8_scalar(&accel, &mut x, &mut v, dt, steps);
                (x.map(Scalar::value), v.map(Scalar::value))
            }
        };
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe { *out = from_state(Vec3::new(r[0], r[1], r[2]), Vec3::new(v[0], v[1], v[2])) };
        SL_OK
    })
}

/// Lambert's problem. See the header.
///
/// # Safety
/// `r1` and `r2` point at initialised `sl_vec3`; `v1` and `v2` are writable for one each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_lambert(
    r1: *const SlVec3,
    r2: *const SlVec3,
    tof: f64,
    mu: f64,
    prograde: bool,
    v1: *mut SlVec3,
    v2: *mut SlVec3,
) -> i32 {
    guard(|| {
        if r1.is_null() || r2.is_null() || v1.is_null() || v2.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_lambert: null pointer");
        }
        // SAFETY: checked non-null just above.
        let (a, b) = unsafe { (*r1, *r2) };
        match astro_lambert(to_vec3(a), to_vec3(b), tof, mu, prograde) {
            Some((va, vb)) => {
                // SAFETY: checked non-null; written only now that the answer exists.
                unsafe {
                    *v1 = from_vec3(va);
                    *v2 = from_vec3(vb);
                }
                SL_OK
            }
            None => fail(
                SL_ERR_DOMAIN,
                "sl_lambert: no transfer arc for those positions and time",
            ),
        }
    })
}

// ── Measurements from the ground ──────────────────────────────────────────────────────────────

/// Range from station to satellite, metres.
///
/// # Safety
/// `sat` and `station_ecef` are null or point at initialised values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_range(sat: *const SlState, station_ecef: *const SlVec3) -> f64 {
    guard_f64(|| {
        if sat.is_null() || station_ecef.is_null() {
            return f64::NAN;
        }
        // SAFETY: checked non-null just above.
        let (s, p) = unsafe { (*sat, *station_ecef) };
        m_range(&to_six(s), &Station::fixed(to_three(p)))
    })
}

/// Range rate, metres per second, positive when opening.
///
/// # Safety
/// `sat` and `station_ecef` are null or point at initialised values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_range_rate(sat: *const SlState, station_ecef: *const SlVec3) -> f64 {
    guard_f64(|| {
        if sat.is_null() || station_ecef.is_null() {
            return f64::NAN;
        }
        // SAFETY: checked non-null just above.
        let (s, p) = unsafe { (*sat, *station_ecef) };
        m_range_rate(&to_six(s), &Station::fixed(to_three(p)))
    })
}

/// Topocentric elevation above the local horizon, radians.
///
/// # Safety
/// `sat` and `station_ecef` are null or point at initialised values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_elevation(
    sat: *const SlState,
    station_ecef: *const SlVec3,
    latitude_rad: f64,
    longitude_rad: f64,
) -> f64 {
    guard_f64(|| {
        if sat.is_null() || station_ecef.is_null() {
            return f64::NAN;
        }
        // SAFETY: checked non-null just above.
        let (s, p) = unsafe { (*sat, *station_ecef) };
        let topo = Topocentric::from_geodetic(to_three(p), latitude_rad, longitude_rad);
        m_elevation(&to_six(s), &topo)
    })
}

/// Topocentric azimuth, radians, from north through east.
///
/// # Safety
/// `sat` and `station_ecef` are null or point at initialised values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_azimuth(
    sat: *const SlState,
    station_ecef: *const SlVec3,
    latitude_rad: f64,
    longitude_rad: f64,
) -> f64 {
    guard_f64(|| {
        if sat.is_null() || station_ecef.is_null() {
            return f64::NAN;
        }
        // SAFETY: checked non-null just above.
        let (s, p) = unsafe { (*sat, *station_ecef) };
        let topo = Topocentric::from_geodetic(to_three(p), latitude_rad, longitude_rad);
        m_azimuth(&to_six(s), &topo)
    })
}

// ── Footprint geometry ────────────────────────────────────────────────────────────────────────

/// The limb look angle, radians.
#[unsafe(no_mangle)]
pub extern "C" fn sl_horizon_look_angle(earth_radius_m: f64, altitude_m: f64) -> f64 {
    guard_f64(|| ev_horizon_look_angle(earth_radius_m, altitude_m))
}

/// Where a great-circle arc from `start` arrives.
#[unsafe(no_mangle)]
pub extern "C" fn sl_ground_point_at(
    start: SlLatLon,
    central_angle_rad: f64,
    azimuth_rad: f64,
) -> SlLatLon {
    let mut result = SlLatLon {
        latitude_rad: f64::NAN,
        longitude_rad: f64::NAN,
    };
    let _ = guard(|| {
        let (latitude_rad, longitude_rad) = ev_ground_point_at(
            start.latitude_rad,
            start.longitude_rad,
            central_angle_rad,
            azimuth_rad,
        );
        result = SlLatLon {
            latitude_rad,
            longitude_rad,
        };
        SL_OK
    });
    result
}

// ── Orbit design ──────────────────────────────────────────────────────────────────────────────

/// The prograde critical inclination, radians.
#[unsafe(no_mangle)]
pub extern "C" fn sl_critical_inclination() -> f64 {
    guard_f64(|| design_critical_inclination().radians())
}

/// The frozen-orbit eccentricity.
#[unsafe(no_mangle)]
pub extern "C" fn sl_frozen_eccentricity(
    semi_major_axis_m: f64,
    inclination_rad: f64,
    earth_radius_m: f64,
    j2: f64,
    j3: f64,
) -> f64 {
    guard_f64(|| {
        design_frozen_eccentricity(
            semi_major_axis_m,
            Angle::from_radians(inclination_rad),
            earth_radius_m,
            j2,
            j3,
        )
    })
}

/// The Walker pattern a tag names, or the refusal that says it names none.
fn walker_pattern_of(who: &str, tag: i32) -> Result<WalkerPattern, i32> {
    match tag {
        WALKER_DELTA => Ok(WalkerPattern::Delta),
        WALKER_STAR => Ok(WalkerPattern::Star),
        other => Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: pattern tag {other} is not an sl_walker_pattern"),
        )),
    }
}

/// A repeat cycle is `revs` revolutions in `days` days, and neither may be zero: zero divides, and
/// the infinity that comes back reads as an orbit right up until something differences two of them.
fn check_repeat_cycle(who: &str, revs: u32, days: u32) -> Result<(), i32> {
    if revs == 0 || days == 0 {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: revs is {revs} and days is {days}, and neither may be zero"),
        ));
    }
    Ok(())
}

/// The repeat-ground-track semi-major axis. See the header.
///
/// # Safety
/// `out` is writable for one double.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)] // the repeat cycle, the orbit shape, and four field constants
pub unsafe extern "C" fn sl_repeat_ground_track_sma(
    revs: u32,
    days: u32,
    inclination_rad: f64,
    eccentricity: f64,
    mu: f64,
    earth_radius_m: f64,
    j2: f64,
    earth_rotation_rate: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_repeat_ground_track_sma: null out pointer",
            );
        }
        if let Err(status) = check_repeat_cycle("sl_repeat_ground_track_sma", revs, days) {
            return status;
        }
        let sma = design_repeat_ground_track_sma(
            revs,
            days,
            eccentricity,
            Angle::from_radians(inclination_rad),
            mu,
            earth_radius_m,
            j2,
            earth_rotation_rate,
        );
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe { *out = sma };
        SL_OK
    })
}

/// The Sun-synchronous inclination. See the header.
///
/// # Safety
/// `out` is writable for one double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sun_synchronous_inclination(
    semi_major_axis_m: f64,
    eccentricity: f64,
    mu: f64,
    earth_radius_m: f64,
    j2: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_sun_synchronous_inclination: null out pointer",
            );
        }
        match design_sun_synchronous_inclination(
            semi_major_axis_m,
            eccentricity,
            mu,
            earth_radius_m,
            j2,
            SUN_SYNCHRONOUS_NODE_RATE,
        ) {
            Some(angle) => {
                // SAFETY: checked non-null; written only now that the answer exists.
                unsafe { *out = angle.radians() };
                SL_OK
            }
            None => fail(
                SL_ERR_DOMAIN,
                "sl_sun_synchronous_inclination: no inclination precesses this orbit a turn a year",
            ),
        }
    })
}

// ── Constellations ────────────────────────────────────────────────────────────────────────────

/// Fill `p` with the defaults documented in the header.
///
/// # Safety
/// `p` is writable for one `sl_walker_params`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_walker_params_init(p: *mut SlWalkerParams) {
    if p.is_null() {
        return;
    }
    // SAFETY: checked non-null just above.
    unsafe {
        *p = SlWalkerParams {
            pattern: WALKER_DELTA,
            inclination_rad: 0.0,
            planes: 0,
            per_plane: 0,
            phasing: 0,
            semi_major_axis_m: 0.0,
            mu: MU_EARTH,
        }
    };
}

/// Fill `p` with the defaults documented in the header.
///
/// # Safety
/// `p` is writable for one `sl_soc_params`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_soc_params_init(p: *mut SlSocParams) {
    if p.is_null() {
        return;
    }
    // SAFETY: checked non-null just above.
    unsafe {
        *p = SlSocParams {
            inclination_rad: 0.0,
            altitude_m: 0.0,
            min_elevation_rad: 0.0,
            earth_radius_m: RE_EARTH,
            mu: MU_EARTH,
            per_plane: 0,
        }
    };
}

/// Earth's gravitational parameter, the default the `_init` functions write. Mirrors
/// `SL_MU_EARTH`.
const MU_EARTH: f64 = 3.986_004_418e14;

/// WGS-84 equatorial radius, the default the `_init` functions write. Mirrors `SL_RE_EARTH`.
const RE_EARTH: f64 = 6_378_137.0;

/// Heliocentric gravitational parameter, IAU 2015 / DE440, m³/s².
const MU_SUN: f64 = 1.327_124_400_18e20;

/// Selenocentric gravitational parameter, m³/s².
const MU_MOON: f64 = 4.902_800_066e12;

/// The defining astronomical unit, metres.
const ASTRONOMICAL_UNIT: f64 = 1.495_978_707e11;

/// Write `values` through the two-call protocol documented in the header.
///
/// Generic over the element because two shapes travel this way — states from the generators, and
/// access windows from the coverage engine — and the protocol is the same for both.
///
/// # Safety
/// `out` is null or writable for `cap` elements; `out_len` is writable for one `usize`.
unsafe fn deliver<T: Copy>(values: &[T], out: *mut T, cap: usize, out_len: *mut usize) -> i32 {
    if out_len.is_null() {
        return fail(SL_ERR_INVALID_ARGUMENT, "null out_len pointer");
    }
    // SAFETY: checked non-null. Written before the capacity test on purpose: the length is what a
    // `SL_ERR_BUFFER_TOO_SMALL` exists to report, and the header documents that exception.
    unsafe { *out_len = values.len() };
    if out.is_null() {
        return SL_OK;
    }
    if cap < values.len() {
        return fail(
            SL_ERR_BUFFER_TOO_SMALL,
            format!(
                "buffer holds {cap} elements, the result needs {}",
                values.len()
            ),
        );
    }
    // SAFETY: `out` is writable for `cap >= values.len()` elements and does not alias `values`,
    // which is a local `Vec` this crate owns.
    unsafe { core::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len()) };
    SL_OK
}

/// Generate a Walker constellation. See the header.
///
/// # Safety
/// `p` points at one initialised `sl_walker_params`; `out` is null or writable for `cap` states;
/// `out_len` is writable for one `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_walker(
    p: *const SlWalkerParams,
    out: *mut SlState,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if p.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_walker: null params pointer");
        }
        // SAFETY: checked non-null just above.
        let p = unsafe { *p };
        let pattern = match walker_pattern_of("sl_walker", p.pattern) {
            Ok(pattern) => pattern,
            Err(status) => return status,
        };
        let walker = Walker {
            pattern,
            inclination: Angle::from_radians(p.inclination_rad),
            planes: p.planes,
            per_plane: p.per_plane,
            phasing: p.phasing,
            semi_major_axis: p.semi_major_axis_m,
            mu: p.mu,
        };
        // Generated even for the sizing call. The alternative -- returning `planes * per_plane` --
        // would be an independent claim about what `generate` emits, and a count that can disagree
        // with the thing it counts is worse than the allocation it saves.
        let mut states = Vec::new();
        walker.generate(|e| {
            let (r, v) = e.to_state();
            states.push(from_state(r, v));
        });
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&states, out, cap, out_len) }
    })
}

/// Build the Rust design from the C parameters, mapping the `per_plane == 0` sentinel.
fn street_of_coverage(p: SlSocParams) -> StreetOfCoverage {
    StreetOfCoverage {
        inclination: Angle::from_radians(p.inclination_rad),
        altitude: p.altitude_m,
        min_elevation: Angle::from_radians(p.min_elevation_rad),
        earth_radius: p.earth_radius_m,
        mu: p.mu,
        per_plane: (p.per_plane != 0).then_some(p.per_plane),
    }
}

/// Solve a Street-of-Coverage geometry. See the header.
///
/// # Safety
/// `p` points at one initialised `sl_soc_params`; `out` is writable for one `sl_soc_geometry`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_street_of_coverage_geometry(
    p: *const SlSocParams,
    out: *mut SlSocGeometry,
) -> i32 {
    guard(|| {
        if p.is_null() || out.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_street_of_coverage_geometry: null pointer",
            );
        }
        // SAFETY: checked non-null just above.
        let p = unsafe { *p };
        match street_of_coverage(p).geometry() {
            Some(g) => {
                // SAFETY: checked non-null; written only now that the answer exists.
                unsafe {
                    *out = SlSocGeometry {
                        planes: g.planes,
                        per_plane: g.per_plane,
                        footprint_half_angle_rad: g.footprint_half_angle,
                        street_half_width_rad: g.street_half_width,
                        raan_step_co_rad: g.raan_step_co,
                        raan_step_counter_rad: g.raan_step_counter,
                        raan_step_rad: g.raan_step,
                        raan_span_rad: g.raan_span,
                        inter_plane_phase_rad: g.inter_plane_phase,
                    }
                };
                SL_OK
            }
            None => fail(
                SL_ERR_DOMAIN,
                "sl_street_of_coverage_geometry: no street tiles this footprint, inclination and \
                 satellite count",
            ),
        }
    })
}

/// Generate a Street-of-Coverage constellation. See the header.
///
/// # Safety
/// `p` points at one initialised `sl_soc_params`; `out` is null or writable for `cap` states;
/// `out_len` is writable for one `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_street_of_coverage(
    p: *const SlSocParams,
    out: *mut SlState,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if p.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_street_of_coverage: null params pointer",
            );
        }
        // SAFETY: checked non-null just above.
        let p = unsafe { *p };
        let soc = street_of_coverage(p);
        // Asked before generating, so an infeasible design is a named domain error rather than the
        // empty list the Python binding returns and leaves the caller to interpret.
        if soc.geometry().is_none() {
            return fail(
                SL_ERR_DOMAIN,
                "sl_street_of_coverage: no street tiles this footprint, inclination and satellite \
                 count",
            );
        }
        let mut states = Vec::new();
        soc.generate(|e| {
            let (r, v) = e.to_state();
            states.push(from_state(r, v));
        });
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&states, out, cap, out_len) }
    })
}

// ── Numerical propagation ─────────────────────────────────────────────────────────────────────

/// `sl_ephemeris`: a dense interpolant of a propagated orbit. Opaque to C.
///
/// One handle over two integrators because a caller wants an orbit, not a method: the tag says
/// which one built it, and every query below reads through it. The alternative — two handle types
/// with two sets of query functions — doubles the C surface to expose a choice the caller already
/// made when they picked the constructor.
pub struct SlEphemeris {
    inner: EphemerisKind,
}

enum EphemerisKind {
    /// Adaptive DOP853 with its own dense output; what the geopotential propagator produces.
    Dop853(Box<Trajectory<6, Dop853Dense<6>>>),
    /// IAS15 with the b-polynomial interpolant; position-only forces, round-off-floor accuracy.
    Ias15(Box<Trajectory<6, Ias15Dense<3>>>),
}

// The same promise `SlSgp4` makes, for the same reason: the header says a handle may be read from
// several threads at once, and across FFI nothing on the C side could be told otherwise.
const _: () = {
    const fn assert_shareable<T: Send + Sync>() {}
    assert_shareable::<SlEphemeris>();
};

impl EphemerisKind {
    fn eval(&self, t: f64) -> [f64; 6] {
        match self {
            Self::Dop853(traj) => traj.eval(t),
            Self::Ias15(traj) => traj.eval(t),
        }
    }

    fn span(&self) -> (f64, f64) {
        match self {
            Self::Dop853(traj) => traj.span(),
            Self::Ias15(traj) => traj.span(),
        }
    }

    fn step_counts(&self) -> (u32, u32) {
        match self {
            Self::Dop853(traj) => traj.step_counts(),
            Self::Ias15(traj) => traj.step_counts(),
        }
    }
}

fn from_six(y: [f64; 6]) -> SlState {
    SlState {
        position: SlVec3 {
            x: y[0],
            y: y[1],
            z: y[2],
        },
        velocity: SlVec3 {
            x: y[3],
            y: y[4],
            z: y[5],
        },
    }
}

/// The first step to try, from the span. A dense integrator adapts from here, so this only has to
/// be finite and the right order of magnitude; a thousandth of the arc is what the Python bridge
/// uses and the two are kept identical so their answers can be compared.
fn initial_step(t1: f64) -> f64 {
    (t1 / 1000.0).abs().max(1.0)
}

/// Hand a freshly built trajectory to the caller as a handle. Written last, as everywhere here, so
/// a failed integration cannot leave a caller holding a pointer to nothing.
///
/// # Safety
/// `out` is null or writable for one pointer.
unsafe fn deliver_ephemeris(inner: EphemerisKind, out: *mut *mut SlEphemeris) -> i32 {
    // SAFETY: the caller states `out` is writable; null was rejected before this is reached.
    unsafe { *out = Box::into_raw(Box::new(SlEphemeris { inner })) };
    SL_OK
}

/// Propagate under point-mass gravity plus J2 with IAS15, keeping the dense interpolant.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `out` is writable for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_j2(
    r#in: *const SlState,
    t1_seconds: f64,
    mu: f64,
    j2: f64,
    earth_radius_m: f64,
    out: *mut *mut SlEphemeris,
) -> i32 {
    guard(|| {
        if r#in.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_ephemeris_j2: null pointer");
        }
        if let Err(status) = check_t1("sl_ephemeris_j2", t1_seconds) {
            return status;
        }
        // SAFETY: checked non-null just above; the caller states it is initialised.
        let state = unsafe { *r#in };
        let r0 = to_three(state.position);
        let v0 = to_three(state.velocity);
        let forces = (
            PointMass { mu },
            J2 {
                mu,
                re: earth_radius_m,
                j2,
            },
        );
        let accel = |r: &[f64; 3]| forces.acceleration::<f64>(0.0, r, &[0.0_f64; 3]);
        let traj = match integrate_ias15_dense::<3, 6, _>(
            &accel,
            &r0,
            &v0,
            0.0,
            t1_seconds,
            initial_step(t1_seconds),
            DEFAULT_MAX_STEPS,
        ) {
            Ok(traj) => traj,
            Err(e) => return fail(SL_ERR_COMPUTE, format!("IAS15 propagation failed: {e:?}")),
        };
        // SAFETY: `out` was checked non-null above.
        unsafe { deliver_ephemeris(EphemerisKind::Ias15(Box::new(traj)), out) }
    })
}

/// Propagate under the embedded tesseral geopotential, evaluated Earth-fixed, with DOP853.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `out` is writable for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_geopotential(
    r#in: *const SlState,
    t1_seconds: f64,
    degree: u32,
    theta0_rad: f64,
    tolerance: f64,
    out: *mut *mut SlEphemeris,
) -> i32 {
    guard(|| {
        if r#in.is_null() || out.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_ephemeris_geopotential: null pointer",
            );
        }
        if let Err(status) = check_arc("sl_ephemeris_geopotential", t1_seconds, tolerance) {
            return status;
        }
        let Some(geo) = GeoPotential::embedded(degree as usize) else {
            return fail(
                SL_ERR_DOMAIN,
                format!("degree {degree} exceeds the embedded model (max {MODEL_MAX_DEGREE})"),
            );
        };
        // SAFETY: checked non-null just above; the caller states it is initialised.
        let state = unsafe { *r#in };
        let sys = Plain(Newtonian(EarthFixed {
            inner: geo,
            omega: spacelab_frames::EARTH_ROTATION_RATE,
            theta0: theta0_rad,
        }));
        let y0 = to_six(state);
        let traj = match integrate_dop853(
            &sys,
            0.0,
            &y0,
            t1_seconds,
            initial_step(t1_seconds),
            Controls::new(Tolerance::new(tolerance, tolerance), 1e-3),
        ) {
            Ok(traj) => traj,
            Err(e) => {
                return fail(SL_ERR_COMPUTE, format!("dense propagation failed: {e:?}"));
            }
        };
        // SAFETY: `out` was checked non-null above.
        unsafe { deliver_ephemeris(EphemerisKind::Dop853(Box::new(traj)), out) }
    })
}

/// Release an ephemeris handle. `sl_ephemeris_close(NULL)` is a no-op.
///
/// # Safety
/// `e` is null, or a handle from `sl_ephemeris_*` that has not been closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_close(e: *mut SlEphemeris) {
    if e.is_null() {
        return;
    }
    // SAFETY: the caller states this came from `Box::into_raw` in `deliver_ephemeris` and has not
    // been freed; reconstructing the Box hands it back to the allocator that made it.
    drop(unsafe { Box::from_raw(e) });
}

/// Every epoch has to lie inside the span the ephemeris covers, which is not ordered: an arc
/// integrated backwards has `t0 > t1`, and a comparison written the obvious way rejects every epoch
/// in it. Ordering the two ends is part of the check, so it belongs with the check.
fn check_within_span(times: &[f64], span: (f64, f64)) -> Result<(), i32> {
    let (t0, t1) = span;
    let (lo, hi) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
    if let Some(&t) = times.iter().find(|&&t| !(t >= lo && t <= hi)) {
        return Err(fail(
            SL_ERR_DOMAIN,
            format!("t = {t} is outside the span [{lo}, {hi}]"),
        ));
    }
    Ok(())
}

/// Interpolate the state at `t_seconds` from the start of the span.
///
/// # Safety
/// `e` is a live handle; `out` is writable for one `sl_state`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_at(
    e: *const SlEphemeris,
    t_seconds: f64,
    out: *mut SlState,
) -> i32 {
    guard(|| {
        if e.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_ephemeris_at: null pointer");
        }
        // SAFETY: checked non-null; the caller states the handle is live.
        let eph = unsafe { &*e };
        if let Err(status) = check_within_span(&[t_seconds], eph.inner.span()) {
            return status;
        }
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe { *out = from_six(eph.inner.eval(t_seconds)) };
        SL_OK
    })
}

/// Interpolate at every epoch in `times_seconds`, writing `n` states.
///
/// One call rather than `n`, for the same reason `sl_sgp4_propagate_batch` exists: the crossing is
/// the expensive part of a per-sample loop from a managed language.
///
/// # Safety
/// `e` is a live handle; `times_seconds` and `out` are readable and writable for `n` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_sample(
    e: *const SlEphemeris,
    times_seconds: *const f64,
    n: usize,
    out: *mut SlState,
) -> i32 {
    guard(|| {
        if e.is_null() || times_seconds.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_ephemeris_sample: null pointer");
        }
        if let Err(status) = check_nonzero("sl_ephemeris_sample", "n", n) {
            return status;
        }
        // SAFETY: checked non-null; the caller states the handle is live.
        let eph = unsafe { &*e };
        // SAFETY: the caller states `times_seconds` is readable for `n` doubles.
        let times = unsafe { core::slice::from_raw_parts(times_seconds, n) };
        // Every epoch is checked before any state is written: a half-filled answer is worse than a
        // refused one, and the caller cannot tell how far the loop got.
        if let Err(status) = check_within_span(times, eph.inner.span()) {
            return status;
        }
        for (i, &t) in times.iter().enumerate() {
            // SAFETY: `out` is writable for `n` states and `i < n`.
            unsafe { *out.add(i) = from_six(eph.inner.eval(t)) };
        }
        SL_OK
    })
}

/// The `[t0, t1]` the interpolant covers, seconds from the start.
///
/// # Safety
/// `e` is a live handle; both out-parameters are writable for one `double`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_span(
    e: *const SlEphemeris,
    out_t0: *mut f64,
    out_t1: *mut f64,
) -> i32 {
    guard(|| {
        if e.is_null() || out_t0.is_null() || out_t1.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_ephemeris_span: null pointer");
        }
        // SAFETY: checked non-null; the caller states the handle is live.
        let (t0, t1) = unsafe { &*e }.inner.span();
        // SAFETY: checked non-null.
        unsafe {
            *out_t0 = t0;
            *out_t1 = t1;
        }
        SL_OK
    })
}

/// How hard the problem was: accepted and rejected step counts.
///
/// # Safety
/// `e` is a live handle; both out-parameters are writable for one `uint32_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_step_counts(
    e: *const SlEphemeris,
    out_accepted: *mut u32,
    out_rejected: *mut u32,
) -> i32 {
    guard(|| {
        if e.is_null() || out_accepted.is_null() || out_rejected.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_ephemeris_step_counts: null pointer",
            );
        }
        // SAFETY: checked non-null; the caller states the handle is live.
        let (accepted, rejected) = unsafe { &*e }.inner.step_counts();
        // SAFETY: checked non-null.
        unsafe {
            *out_accepted = accepted;
            *out_rejected = rejected;
        }
        SL_OK
    })
}

/// The tesseral geopotential acceleration at a body-fixed position.
///
/// # Safety
/// `r_body_fixed` points at one initialised `sl_vec3`; `out` is writable for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_geopotential_acceleration(
    degree: u32,
    r_body_fixed: *const SlVec3,
    out: *mut SlVec3,
) -> i32 {
    guard(|| {
        if r_body_fixed.is_null() || out.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_geopotential_acceleration: null pointer",
            );
        }
        let Some(geo) = GeoPotential::embedded(degree as usize) else {
            return fail(
                SL_ERR_DOMAIN,
                format!("degree {degree} exceeds the embedded model (max {MODEL_MAX_DEGREE})"),
            );
        };
        // SAFETY: checked non-null; the caller states it is initialised.
        let r = to_three(unsafe { *r_body_fixed });
        let a = geo.acceleration(0.0, &r, &[0.0_f64; 3]);
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe {
            *out = SlVec3 {
                x: a[0],
                y: a[1],
                z: a[2],
            };
        }
        SL_OK
    })
}

/// The CR3BP is a two-body system, so its mass ratio is a proper fraction: at 0 or 1 one of the
/// primaries has vanished and the model is not the one the caller asked for.
fn check_mass_ratio(who: &str, mass_ratio: f64) -> Result<(), i32> {
    if !(mass_ratio.is_finite() && mass_ratio > 0.0 && mass_ratio < 1.0) {
        return Err(fail(
            SL_ERR_DOMAIN,
            format!("{who}: the mass ratio is {mass_ratio}, and must lie in (0, 1)"),
        ));
    }
    Ok(())
}

/// A libration point's number as the crate's zero-based index into the collinear points.
///
/// `li` is the point's name -- L1 is 1 -- where the crate function takes an index. The domain calls
/// it L1; an API that made the caller write 0 for it would be a translation waiting to be got wrong,
/// so the translation happens here, once. Deliberately unlike the Python binding, which passes the
/// index through; the same kind of divergence as radians against its degrees.
fn libration_index(who: &str, li: u32) -> Result<u32, i32> {
    li.checked_sub(1).ok_or_else(|| {
        fail(
            SL_ERR_DOMAIN,
            format!("{who}: libration points are numbered from 1"),
        )
    })
}

/// The five libration points of the circular restricted three-body problem, synodic and
/// nondimensional, in the order L1, L2, L3, L4, L5.
///
/// # Safety
/// `out` is writable for five `sl_vec3`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_libration_points(mass_ratio: f64, out: *mut SlVec3) -> i32 {
    guard(|| {
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_libration_points: null out");
        }
        if let Err(status) = check_mass_ratio("sl_libration_points", mass_ratio) {
            return status;
        }
        let points = Cr3bp { mu: mass_ratio }.libration_points();
        for (i, p) in points.iter().enumerate() {
            // SAFETY: the caller states `out` is writable for five, and `i < 5`.
            unsafe { *out.add(i) = from_vec3(*p) };
        }
        SL_OK
    })
}

/// Richardson's third-order halo-orbit initial guess about libration point `li` (1, 2 or 3).
///
/// # Safety
/// `out` is writable for one `sl_state`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_richardson_halo(
    li: u32,
    amplitude_z: f64,
    northern: bool,
    mass_ratio: f64,
    out: *mut SlState,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_richardson_halo: null out");
        }
        let index = match libration_index("sl_richardson_halo", li) {
            Ok(index) => index,
            Err(status) => return status,
        };
        let Some(y) = richardson_halo_guess(
            Cr3bp { mu: mass_ratio },
            index as usize,
            amplitude_z,
            northern,
        ) else {
            return fail(
                SL_ERR_DOMAIN,
                format!(
                    "no halo guess for L{li} at amplitude {amplitude_z}; the third-order theory \
                     covers L1 and L2 only"
                ),
            );
        };
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe { *out = from_six(y) };
        SL_OK
    })
}

// ── Semi-analytical propagation (DSST) ────────────────────────────────────────────────────────

/// The most zonal harmonics `sl_dsst_params` carries, J2 through J17.
///
/// An inline array rather than a pointer and a count: it makes the parameter block a value the
/// caller puts on the stack and copies, with no lifetime to outlive and no allocation to own, and
/// DSST's zonal theory is a low-degree one — the default stops at J5 and Orekit's at J6. A field
/// that wanted degree 120 would be the tesseral model, which is a different entry point.
pub const SL_DSST_MAX_ZONAL: usize = 16;

/// `sl_dsst_params`: the field, the spacecraft and the epoch a DSST propagation runs against.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlDsstParams {
    /// Central-body gravitational parameter, m³/s².
    pub mu: f64,
    /// Central-body equatorial radius, metres.
    pub earth_radius_m: f64,
    /// The initial epoch on the J2000 timeline, seconds — where the Sun and Moon are.
    pub epoch_j2000_seconds: f64,
    /// Radiation-pressure coefficient `Cr`, dimensionless.
    pub reflectivity: f64,
    /// Cross-section over mass, m²/kg.
    pub area_over_mass: f64,
    /// Solar gravitational parameter, m³/s².
    pub mu_sun: f64,
    /// Lunar gravitational parameter, m³/s².
    pub mu_moon: f64,
    /// `[J2, J3, …]`, indexed from degree 2; only the first `zonal_count` are read.
    pub zonal: [f64; SL_DSST_MAX_ZONAL],
    /// How many of `zonal` carry a coefficient. At least 1, at most `SL_DSST_MAX_ZONAL`.
    pub zonal_count: u32,
}

/// Fill `p` with the defaults documented in the header.
///
/// # Safety
/// `p` is writable for one `sl_dsst_params`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_dsst_params_init(p: *mut SlDsstParams) {
    if p.is_null() {
        return;
    }
    let mut zonal = [0.0_f64; SL_DSST_MAX_ZONAL];
    // EIGEN-6S normalised zonals through J5, the same four the Python bridge defaults to, so the
    // two bindings answer identically on a call that names no field.
    zonal[..4].copy_from_slice(&[
        1.082_626_68e-3,
        -2.532_656_49e-6,
        -1.619_621_59e-6,
        -2.272_960_83e-7,
    ]);
    // SAFETY: checked non-null just above.
    unsafe {
        *p = SlDsstParams {
            mu: 3.986_004_415e14,
            earth_radius_m: 6_378_136.3,
            epoch_j2000_seconds: 0.0,
            reflectivity: 1.5,
            area_over_mass: 0.02,
            mu_sun: 1.327_124_400_18e20,
            mu_moon: 4.902_800_066e12,
            zonal,
            zonal_count: 4,
        }
    };
}

/// Read `params`, validate it, and build the zonal field — or name what is wrong with it.
fn dsst_zonal_field(who: &str, p: &SlDsstParams) -> Result<DsstZonal, i32> {
    if !(p.mu.is_finite() && p.mu > 0.0) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: mu must be finite and positive"),
        ));
    }
    if !(p.earth_radius_m.is_finite() && p.earth_radius_m > 0.0) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: earth_radius_m must be finite and positive"),
        ));
    }
    let count = p.zonal_count as usize;
    if count == 0 || count > SL_DSST_MAX_ZONAL {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!(
                "{who}: zonal_count is {}, and must lie in 1..={SL_DSST_MAX_ZONAL}",
                p.zonal_count
            ),
        ));
    }
    let j = &p.zonal[..count];
    if let Some(i) = j.iter().position(|v| !v.is_finite()) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: J{} is not finite", i + 2),
        ));
    }
    Ok(DsstZonal::from_j(p.mu, p.earth_radius_m, j))
}

/// The remaining scalars the combined propagator reads, checked once for all four entry points.
fn dsst_check_spacecraft(who: &str, p: &SlDsstParams) -> Result<(), i32> {
    for (name, value) in [
        ("epoch_j2000_seconds", p.epoch_j2000_seconds),
        ("reflectivity", p.reflectivity),
        ("area_over_mass", p.area_over_mass),
        ("mu_sun", p.mu_sun),
        ("mu_moon", p.mu_moon),
    ] {
        if !value.is_finite() {
            return Err(fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{who}: {name} must be finite"),
            ));
        }
    }
    Ok(())
}

/// What one DSST call is asked for: where the initial state is, which epochs to answer at, and
/// where the answers go.
///
/// Four pointers that always travel together, so they travel as one value — and it is those four
/// that push the combined entry point past what a readable argument list holds.
///
/// The pointers are the caller's: `initial` is readable for six doubles, `times_seconds` for `n`,
/// and `out` is writable for `6 * n`.
#[derive(Clone, Copy)]
struct DsstRequest {
    initial: *const f64,
    times_seconds: *const f64,
    n: usize,
    out: *mut f64,
}

/// The shared body of the four DSST entry points: validate the request, then write one equinoctial
/// sextuple per requested epoch.
///
/// The state is *mean* equinoctial `[a, k, h, q, p, λ]` on the way in, which is the only form DSST
/// takes; what comes back is mean or osculating depending on which `step` the caller passed.
///
/// # Safety
/// The pointers in `req` satisfy the contract documented on [`DsstRequest`].
/// A mean equinoctial state has to be finite, and its first element is a semi-major axis, which is
/// a length. A non-positive `a` is not a small orbit, it is not an orbit.
fn check_mean_equinoctial(who: &str, y0: &[f64; 6]) -> Result<(), i32> {
    check_all_finite(who, "the initial mean equinoctial state", y0)?;
    if y0[0] <= 0.0 {
        return Err(fail(
            SL_ERR_DOMAIN,
            format!(
                "{who}: the semi-major axis is {}, and must be positive",
                y0[0]
            ),
        ));
    }
    Ok(())
}

unsafe fn dsst_sample(
    who: &str,
    req: DsstRequest,
    step: impl Fn([f64; 6], f64) -> [f64; 6],
) -> i32 {
    let DsstRequest {
        initial,
        times_seconds,
        n,
        out,
    } = req;
    if initial.is_null() || times_seconds.is_null() || out.is_null() {
        return fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: null pointer"));
    }
    if let Err(status) = check_nonzero(who, "n", n) {
        return status;
    }
    let mut y0 = [0.0_f64; 6];
    // SAFETY: the caller states `initial` is readable for six doubles.
    y0.copy_from_slice(unsafe { core::slice::from_raw_parts(initial, 6) });
    if let Err(status) = check_mean_equinoctial(who, &y0) {
        return status;
    }
    // SAFETY: the caller states `times_seconds` is readable for `n` doubles.
    let times = unsafe { core::slice::from_raw_parts(times_seconds, n) };
    // Every epoch is checked before any of them is propagated, for the same reason
    // `sl_ephemeris_sample` checks its span up front: a half-filled buffer is worse than a refused
    // one, because the caller has no way to learn how far the loop got.
    if let Err(status) = check_all_finite(who, "times_seconds", times) {
        return status;
    }
    for (i, &t) in times.iter().enumerate() {
        let y = step(y0, t);
        // SAFETY: the caller states `out` is writable for `6 * n` doubles, and `i < n`.
        unsafe { core::ptr::copy_nonoverlapping(y.as_ptr(), out.add(i * 6), 6) };
    }
    SL_OK
}

/// Which elements a DSST call answers in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DsstElements {
    /// The orbit-averaged drift alone.
    Mean,
    /// The drift plus every contribution's short-period correction.
    Osculating,
}

/// Build the combined propagator over the four conservative contributions and sample it.
///
/// The tesseral field is deliberately zero. A tesseral mean rate is identically zero away from a
/// resonance — that is, for every orbit except a repeat-ground-track one — so this is the whole
/// conservative mean rate for the ordinary LEO/MEO/HEO case, and carrying a 120-degree field
/// across the C boundary to add nothing to it would be a cost with no answer attached.
///
/// # Safety
/// As [`dsst_sample`], whose pointer contract this forwards verbatim.
unsafe fn dsst_combined_sample(
    who: &str,
    p: &SlDsstParams,
    zonal: &DsstZonal,
    elements: DsstElements,
    req: DsstRequest,
) -> i32 {
    let sun_tb = DsstThirdBody::new(p.mu, p.mu_sun);
    let moon_tb = DsstThirdBody::new(p.mu, p.mu_moon);
    let srp =
        DsstSolarRadiationPressure::new(p.reflectivity, p.area_over_mass, p.mu, p.earth_radius_m);
    let tesseral = DsstTesseral::new(
        p.mu,
        p.earth_radius_m,
        spacelab_frames::EARTH_ROTATION_RATE,
        std::vec![std::vec![0.0; 3]; 3],
        std::vec![std::vec![0.0; 3]; 3],
    );
    let sun = AnalyticSun {
        epoch_j2000_seconds: p.epoch_j2000_seconds,
    };
    let moon = AnalyticMoon {
        epoch_j2000_seconds: p.epoch_j2000_seconds,
    };
    let prop = DsstMeanPropagator {
        zonal,
        sun: &sun_tb,
        moon: &moon_tb,
        srp: &srp,
        tesseral: &tesseral,
        ephemeris: |t: f64| (sun.position(t), moon.position(t)),
        gmst: |_t: f64| 0.0, // unused while the tesseral field is zero
        mu: p.mu,
    };
    // SAFETY: forwarded verbatim; this function's own contract is the one being relied on.
    unsafe {
        dsst_sample(who, req, |y0, t| match elements {
            DsstElements::Mean => prop.propagate(y0, t),
            DsstElements::Osculating => prop.propagate_osculating(y0, t),
        })
    }
}

/// Mean-element propagation under the zonal geopotential alone, `6 * n` doubles out.
///
/// # Safety
/// `params` points at one initialised `sl_dsst_params`; `initial` is readable for six doubles,
/// `times_seconds` for `n`, and `out` writable for `6 * n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_dsst_zonal_mean(
    params: *const SlDsstParams,
    initial: *const f64,
    times_seconds: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_dsst_zonal_mean";
        if params.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null params"));
        }
        let req = DsstRequest {
            initial,
            times_seconds,
            n,
            out,
        };
        // SAFETY: checked non-null; the caller states it is initialised.
        let zonal = match dsst_zonal_field(WHO, unsafe { &*params }) {
            Ok(zonal) => zonal,
            Err(status) => return status,
        };
        // SAFETY: forwarded verbatim; this function's own contract is the one being relied on.
        unsafe { dsst_sample(WHO, req, |y0, t| zonal.propagate_mean(y0, t)) }
    })
}

/// Osculating propagation under the zonal geopotential alone: the mean drift plus the analytic
/// short-period corrections.
///
/// # Safety
/// As [`sl_dsst_zonal_mean`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_dsst_zonal_osculating(
    params: *const SlDsstParams,
    initial: *const f64,
    times_seconds: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_dsst_zonal_osculating";
        if params.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null params"));
        }
        let req = DsstRequest {
            initial,
            times_seconds,
            n,
            out,
        };
        // SAFETY: checked non-null; the caller states it is initialised.
        let zonal = match dsst_zonal_field(WHO, unsafe { &*params }) {
            Ok(zonal) => zonal,
            Err(status) => return status,
        };
        // SAFETY: forwarded verbatim.
        unsafe { dsst_sample(WHO, req, |y0, t| zonal.propagate_osculating(y0, t)) }
    })
}

/// Mean-element propagation under zonal gravity, Sun and Moon third-body attraction, and solar
/// radiation pressure summed into one mean rate.
///
/// # Safety
/// As [`sl_dsst_zonal_mean`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_dsst_mean(
    params: *const SlDsstParams,
    initial: *const f64,
    times_seconds: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_dsst_mean";
        if params.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null params"));
        }
        let req = DsstRequest {
            initial,
            times_seconds,
            n,
            out,
        };
        // SAFETY: checked non-null; the caller states it is initialised.
        let p = unsafe { &*params };
        let zonal = match dsst_zonal_field(WHO, p) {
            Ok(zonal) => zonal,
            Err(status) => return status,
        };
        if let Err(status) = dsst_check_spacecraft(WHO, p) {
            return status;
        }
        // SAFETY: forwarded verbatim.
        unsafe { dsst_combined_sample(WHO, p, &zonal, DsstElements::Mean, req) }
    })
}

/// The same four contributions, returning osculating rather than mean elements: every one of them
/// adds its short-period correction.
///
/// # Safety
/// As [`sl_dsst_zonal_mean`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_dsst_osculating(
    params: *const SlDsstParams,
    initial: *const f64,
    times_seconds: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_dsst_osculating";
        if params.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null params"));
        }
        let req = DsstRequest {
            initial,
            times_seconds,
            n,
            out,
        };
        // SAFETY: checked non-null; the caller states it is initialised.
        let p = unsafe { &*params };
        let zonal = match dsst_zonal_field(WHO, p) {
            Ok(zonal) => zonal,
            Err(status) => return status,
        };
        if let Err(status) = dsst_check_spacecraft(WHO, p) {
            return status;
        }
        // SAFETY: forwarded verbatim.
        unsafe { dsst_combined_sample(WHO, p, &zonal, DsstElements::Osculating, req) }
    })
}

// ── Estimation ────────────────────────────────────────────────────────────────────────────────

/// `sl_residual_row`: one scalar observation, linearised at the current iterate.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlResidualRow {
    /// `∂residual/∂x`, six long.
    pub jacobian: [f64; 6],
    /// `computed(x) − observed`, in the measurement's own units.
    pub residual: f64,
    /// `1/σ²` for this observation.
    pub weight: f64,
}

/// The callback `sl_gauss_newton` calls at each iterate: fill `row_count` rows, return `SL_OK`.
///
/// A fixed row count rather than a growable one, because an orbit determination has the
/// observations it has: the arc is fixed before the solve starts, and a model that changed its
/// measurement count between iterations would be minimising a different problem each time. It also
/// means the buffer is allocated once, by this library, and the callback never allocates.
pub type SlResidualFn = Option<
    unsafe extern "C" fn(
        user_data: *mut c_void,
        state: *const f64,
        rows: *mut SlResidualRow,
        row_count: usize,
    ) -> i32,
>;

/// `sl_gauss_newton_options`: the stopping criteria and damping schedule.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlGaussNewtonOptions {
    /// Give up after this many iterations.
    pub max_iterations: u32,
    /// Converged when the weighted cost falls by less than this fraction in a step.
    pub cost_tolerance: f64,
    /// Converged when the correction, relative to the state norm, is below this.
    pub step_tolerance: f64,
    /// Initial Levenberg–Marquardt damping; `0.0` is pure Gauss–Newton.
    pub initial_lambda: f64,
    /// Converged once the weighted cost reaches this absolute value.
    pub absolute_cost_floor: f64,
}

/// `sl_estimate`: a solved state, its formal covariance, and what the solve cost.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlEstimate {
    /// The solution.
    pub state: [f64; 6],
    /// The 6×6 formal error covariance, row-major; the square roots of its diagonal are the 1σ
    /// uncertainties.
    pub covariance: [f64; 36],
    /// The final weighted sum of squared residuals.
    pub cost: f64,
    /// Iterations taken.
    pub iterations: u32,
    /// Whether a convergence criterion was met, as against the iteration cap being reached.
    pub converged: bool,
}

/// Fill `o` with the defaults documented in the header.
///
/// # Safety
/// `o` is writable for one `sl_gauss_newton_options`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_gauss_newton_options_init(o: *mut SlGaussNewtonOptions) {
    if o.is_null() {
        return;
    }
    // Read from the Rust defaults rather than written down again: two copies of a tuning schedule
    // drift, and the C caller who takes the defaults should be solving the same problem the Rust
    // caller is.
    let d = GaussNewton::default();
    // SAFETY: checked non-null just above.
    unsafe {
        *o = SlGaussNewtonOptions {
            max_iterations: d.max_iterations,
            cost_tolerance: d.cost_tolerance,
            step_tolerance: d.step_tolerance,
            initial_lambda: d.initial_lambda,
            absolute_cost_floor: d.absolute_cost_floor,
        }
    };
}

/// The `ResidualModel` a C callback becomes.
///
/// `status` is the callback's refusal, held until `solve` returns: the trait has no way to report
/// one mid-iteration, and abandoning the solve from inside a callback would mean unwinding through
/// C, which is the one thing this whole boundary exists to prevent.
struct CallbackModel {
    call: unsafe extern "C" fn(*mut c_void, *const f64, *mut SlResidualRow, usize) -> i32,
    user_data: *mut c_void,
    rows: RefCell<Vec<SlResidualRow>>,
    status: Cell<i32>,
}

impl ResidualModel<6> for CallbackModel {
    fn residuals(&self, x: &[f64; 6], emit: &mut dyn FnMut(&[f64; 6], f64, f64)) {
        if self.status.get() != SL_OK {
            return;
        }
        let mut rows = self.rows.borrow_mut();
        // SAFETY: `rows` is a live allocation of exactly the length being passed, and `x` is six
        // readable doubles. That the callback writes only inside them is the caller's half of the
        // contract, and is the reason the header documents the length as the one it was given.
        let code =
            unsafe { (self.call)(self.user_data, x.as_ptr(), rows.as_mut_ptr(), rows.len()) };
        if code != SL_OK {
            self.status.set(fail(
                SL_ERR_COMPUTE,
                format!("the residual model refused at this iterate, returning {code}"),
            ));
            return;
        }
        for row in rows.iter() {
            emit(&row.jacobian, row.residual, row.weight);
        }
    }
}

/// The solver's tuning: an iteration budget that has to allow at least one iteration, and four
/// thresholds that are compared against a cost, a step and a damping factor. Non-negative rather
/// than positive: zero is how a caller says "never stop for this reason", which is a choice.
fn check_gauss_newton_options(who: &str, o: &SlGaussNewtonOptions) -> Result<(), i32> {
    check_nonzero(who, "max_iterations", o.max_iterations)?;
    for (name, value) in [
        ("cost_tolerance", o.cost_tolerance),
        ("step_tolerance", o.step_tolerance),
        ("initial_lambda", o.initial_lambda),
        ("absolute_cost_floor", o.absolute_cost_floor),
    ] {
        check_nonnegative(who, name, value)?;
    }
    Ok(())
}

/// Gauss–Newton / Levenberg–Marquardt least squares over a six-state.
///
/// `model` is called once per iterate to linearise the arc; see `sl_residual_fn`. `options` may be
/// null, which takes the defaults.
///
/// # Safety
/// `x0` is readable for six doubles; `out` is writable for one `sl_estimate`; `options` is null or
/// points at one initialised `sl_gauss_newton_options`. `model` is called with `user_data`
/// unchanged and must not unwind.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_gauss_newton(
    options: *const SlGaussNewtonOptions,
    x0: *const f64,
    model: SlResidualFn,
    user_data: *mut c_void,
    row_count: usize,
    out: *mut SlEstimate,
) -> i32 {
    guard(|| {
        if x0.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_gauss_newton: null pointer");
        }
        let Some(call) = model else {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_gauss_newton: null model");
        };
        if let Err(status) = check_nonzero("sl_gauss_newton", "row_count", row_count) {
            return status;
        }
        let mut start = [0.0_f64; 6];
        // SAFETY: the caller states `x0` is readable for six doubles.
        start.copy_from_slice(unsafe { core::slice::from_raw_parts(x0, 6) });
        if let Err(status) = check_all_finite("sl_gauss_newton", "the initial state", &start) {
            return status;
        }

        let mut solver = GaussNewton::default();
        if !options.is_null() {
            // SAFETY: checked non-null; the caller states it is initialised.
            let o = unsafe { *options };
            if let Err(status) = check_gauss_newton_options("sl_gauss_newton", &o) {
                return status;
            }
            solver = GaussNewton {
                max_iterations: o.max_iterations,
                cost_tolerance: o.cost_tolerance,
                step_tolerance: o.step_tolerance,
                initial_lambda: o.initial_lambda,
                absolute_cost_floor: o.absolute_cost_floor,
            };
        }

        let model = CallbackModel {
            call,
            user_data,
            rows: RefCell::new(std::vec![
                SlResidualRow {
                    jacobian: [0.0; 6],
                    residual: 0.0,
                    weight: 0.0,
                };
                row_count
            ]),
            status: Cell::new(SL_OK),
        };
        let solved = solver.solve(&model, &start);
        // The callback's refusal outranks whatever the solver made of the rows it never got.
        if model.status.get() != SL_OK {
            return model.status.get();
        }
        let (x, cov, report) = match solved {
            Ok(solved) => solved,
            Err(e) => return fail(SL_ERR_DOMAIN, format!("{e}")),
        };
        let mut covariance = [0.0_f64; 36];
        for (i, row) in cov.iter().enumerate() {
            covariance[i * 6..(i + 1) * 6].copy_from_slice(row);
        }
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe {
            *out = SlEstimate {
                state: x,
                covariance,
                cost: report.cost,
                iterations: report.iterations,
                converged: report.converged,
            }
        };
        SL_OK
    })
}

/// `sl_srif`: a square-root information filter over a six-state. Opaque to C.
pub struct SlSrif {
    inner: Srif<6>,
}

// The header promises a handle is shareable; `Srif` is plain arithmetic over arrays, and this stops
// compiling the day that stops being true. See the note on `SlSgp4`.
const _: () = {
    const fn assert_shareable<T: Send + Sync>() {}
    assert_shareable::<SlSrif>();
};

/// A prior is a state and the square root of its information, and a filter seeded with a non-finite
/// one has no recoverable state afterwards: the square-root update propagates the NaN into every
/// row folded in after it.
fn check_srif_prior(who: &str, state: &[f64; 6], information_sqrt: &[f64]) -> Result<(), i32> {
    check_all_finite(who, "x0", state)?;
    check_all_finite(who, "information_sqrt", information_sqrt)
}

/// A measurement row is its Jacobian, the residual it explains and the weight it carries. The
/// weight may be zero -- that is how a caller mutes a row without removing it -- but not negative,
/// which would subtract information the filter never had.
fn check_srif_row(who: &str, jacobian: &[f64; 6], z: f64, weight: f64) -> Result<(), i32> {
    check_all_finite(who, "jacobian", jacobian)?;
    check_finite(who, "z", z)?;
    if !(weight.is_finite() && weight >= 0.0) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: weight is {weight}, and must be finite and non-negative"),
        ));
    }
    Ok(())
}

/// Open a filter, either empty or seeded with a prior.
///
/// `x0` and `information_sqrt` are both null for an empty filter — one with no information, which
/// cannot be solved until enough rows have been folded in — or both non-null for a prior, where
/// `information_sqrt` is the upper-triangular `chol(P₀⁻¹)`, row-major. Passing one and not the
/// other is a mistake rather than a shorthand: a prior state with no prior information says
/// nothing, and prior information about no state has nowhere to sit.
///
/// # Safety
/// `x0` is null or readable for six doubles; `information_sqrt` is null or readable for 36; `out`
/// is writable for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_srif_open(
    x0: *const f64,
    information_sqrt: *const f64,
    out: *mut *mut SlSrif,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_srif_open: null out");
        }
        if x0.is_null() != information_sqrt.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_srif_open: a prior needs both x0 and information_sqrt, or neither",
            );
        }
        let inner = if x0.is_null() {
            Srif::<6>::new()
        } else {
            let mut state = [0.0_f64; 6];
            // SAFETY: checked non-null; the caller states it is readable for six doubles.
            state.copy_from_slice(unsafe { core::slice::from_raw_parts(x0, 6) });
            // SAFETY: checked non-null; the caller states it is readable for 36 doubles.
            let flat = unsafe { core::slice::from_raw_parts(information_sqrt, 36) };
            if let Err(status) = check_srif_prior("sl_srif_open", &state, flat) {
                return status;
            }
            let mut r0 = [[0.0_f64; 6]; 6];
            for (i, row) in r0.iter_mut().enumerate() {
                row.copy_from_slice(&flat[i * 6..(i + 1) * 6]);
            }
            Srif::<6>::with_prior(&state, &r0)
        };
        // SAFETY: checked non-null; written last, so a rejected prior leaves the caller's pointer
        // alone rather than pointing at nothing.
        unsafe { *out = Box::into_raw(Box::new(SlSrif { inner })) };
        SL_OK
    })
}

/// Release a filter. `sl_srif_close(NULL)` is a no-op.
///
/// # Safety
/// `f` is null, or a handle from `sl_srif_open` that has not been closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_srif_close(f: *mut SlSrif) {
    if f.is_null() {
        return;
    }
    // SAFETY: the caller states this came from `Box::into_raw` in `sl_srif_open` and has not been
    // freed; reconstructing the Box hands it back to the allocator that made it.
    drop(unsafe { Box::from_raw(f) });
}

/// Fold one scalar measurement `z = hᵀx + noise` with weight `1/σ²` into the filter.
///
/// # Safety
/// `f` is a live handle; `jacobian` is readable for six doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_srif_add_row(
    f: *mut SlSrif,
    jacobian: *const f64,
    z: f64,
    weight: f64,
) -> i32 {
    guard(|| {
        if f.is_null() || jacobian.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_srif_add_row: null pointer");
        }
        let mut h = [0.0_f64; 6];
        // SAFETY: checked non-null; the caller states it is readable for six doubles.
        h.copy_from_slice(unsafe { core::slice::from_raw_parts(jacobian, 6) });
        if let Err(status) = check_srif_row("sl_srif_add_row", &h, z, weight) {
            return status;
        }
        // SAFETY: checked non-null; the caller states the handle is live and is not being used from
        // another thread at the same time, which is what `sl_srif *` rather than `const` says.
        unsafe { &mut *f }.inner.add_row(&h, z, weight);
        SL_OK
    })
}

/// The state the folded rows imply. Fails while the rows leave a direction unobservable.
///
/// # Safety
/// `f` is a live handle; `out` is writable for six doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_srif_solve(f: *const SlSrif, out: *mut f64) -> i32 {
    guard(|| {
        if f.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_srif_solve: null pointer");
        }
        // SAFETY: checked non-null; the caller states the handle is live.
        match unsafe { &*f }.inner.solve() {
            Ok(x) => {
                // SAFETY: `out` is writable for six doubles; written only now that it exists.
                unsafe { core::ptr::copy_nonoverlapping(x.as_ptr(), out, 6) };
                SL_OK
            }
            Err(e) => fail(SL_ERR_DOMAIN, format!("{e}")),
        }
    })
}

/// The 6×6 formal covariance `(RᵀR)⁻¹`, row-major. Fails on the same rank condition as the solve.
///
/// # Safety
/// `f` is a live handle; `out` is writable for 36 doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_srif_covariance(f: *const SlSrif, out: *mut f64) -> i32 {
    guard(|| {
        if f.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_srif_covariance: null pointer");
        }
        // SAFETY: checked non-null; the caller states the handle is live.
        match unsafe { &*f }.inner.covariance() {
            Ok(cov) => {
                for (i, row) in cov.iter().enumerate() {
                    // SAFETY: `out` is writable for 36 doubles and `i < 6`.
                    unsafe { core::ptr::copy_nonoverlapping(row.as_ptr(), out.add(i * 6), 6) };
                }
                SL_OK
            }
            Err(e) => fail(SL_ERR_DOMAIN, format!("{e}")),
        }
    })
}

/// How well the rows agreed: the post-fit sum of squares rotated past the state, and the row count.
///
/// # Safety
/// `f` is a live handle; both out-parameters are writable for one element.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_srif_residual_stats(
    f: *const SlSrif,
    out_sum_of_squares: *mut f64,
    out_rows: *mut u32,
) -> i32 {
    guard(|| {
        if f.is_null() || out_sum_of_squares.is_null() || out_rows.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_srif_residual_stats: null pointer",
            );
        }
        // SAFETY: checked non-null; the caller states the handle is live.
        let (sum_of_squares, rows) = unsafe { &*f }.inner.residual_stats();
        // SAFETY: checked non-null.
        unsafe {
            *out_sum_of_squares = sum_of_squares;
            *out_rows = rows;
        }
        SL_OK
    })
}

// ── Constellation families ────────────────────────────────────────────────────────────────────

/// `sl_flower_params`: the Flower constellation design (Mortari et al. 2004).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlFlowerParams {
    /// Petals `N_p` of the repeating ground track.
    pub petals: u32,
    /// Days `N_d` the track takes to close.
    pub days: u32,
    /// Spacecraft phased along the common track.
    pub satellites: u32,
    /// Eccentricity of the shared orbit.
    pub eccentricity: f64,
    /// Inclination, radians.
    pub inclination_rad: f64,
    /// Argument of periapsis, radians.
    pub arg_periapsis_rad: f64,
    /// Reference right ascension, radians.
    pub raan_rad: f64,
    /// Reference mean anomaly, radians.
    pub mean_anomaly_rad: f64,
    /// The rotating frame's rate, rad/s.
    pub frame_rate_rad_s: f64,
    /// Gravitational parameter, m³/s².
    pub mu: f64,
}

/// `sl_lattice_params`: the 2-D Lattice Flower design — the integer lattice that unifies Walker
/// and Flower.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlLatticeParams {
    /// Right-ascension slots `N_o`.
    pub orbits: u32,
    /// Satellites in each slot, `N_so`.
    pub sats_per_orbit: u32,
    /// Configuration number `N_c`.
    pub config: u32,
    /// Semi-major axis of the shared orbit, metres.
    pub semi_major_axis_m: f64,
    /// Eccentricity of the shared orbit.
    pub eccentricity: f64,
    /// Inclination, radians.
    pub inclination_rad: f64,
    /// Argument of periapsis, radians.
    pub arg_periapsis_rad: f64,
    /// Right ascension of the first slot, radians.
    pub raan0_rad: f64,
    /// Gravitational parameter, m³/s².
    pub mu: f64,
}

/// `sl_draim_params`: Draim's four-satellite continuous-global-coverage design (Draim 1987).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlDraimParams {
    /// Orbital period, seconds.
    pub period_s: f64,
    /// Eccentricity.
    pub eccentricity: f64,
    /// Inclination, radians.
    pub inclination_rad: f64,
    /// Right ascension of the first satellite, radians.
    pub raan0_rad: f64,
    /// Gravitational parameter, m³/s².
    pub mu: f64,
}

/// The inclination of the Draim patent's worked example, 33°, in radians. Written to full double
/// precision rather than as `33.0_f64.to_radians()` so that the C and Python defaults are the same
/// bits: the two would agree to the last place anyway, but "would" is not a test.
const DRAIM_INCLINATION_RAD: f64 = 0.575_958_653_158_128_8;

/// Fill `p` with the defaults documented in the header.
///
/// # Safety
/// `p` is writable for one `sl_flower_params`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_flower_params_init(p: *mut SlFlowerParams) {
    if p.is_null() {
        return;
    }
    // SAFETY: checked non-null just above.
    unsafe {
        *p = SlFlowerParams {
            petals: 0,
            days: 0,
            satellites: 0,
            eccentricity: 0.0,
            inclination_rad: 0.0,
            arg_periapsis_rad: 0.0,
            raan_rad: 0.0,
            mean_anomaly_rad: 0.0,
            frame_rate_rad_s: 7.292_115_146_5e-5,
            mu: MU_EARTH,
        }
    };
}

/// Fill `p` with the defaults documented in the header.
///
/// # Safety
/// `p` is writable for one `sl_lattice_params`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_lattice_params_init(p: *mut SlLatticeParams) {
    if p.is_null() {
        return;
    }
    // SAFETY: checked non-null just above.
    unsafe {
        *p = SlLatticeParams {
            orbits: 0,
            sats_per_orbit: 0,
            config: 0,
            semi_major_axis_m: 0.0,
            eccentricity: 0.0,
            inclination_rad: 0.0,
            arg_periapsis_rad: 0.0,
            raan0_rad: 0.0,
            mu: MU_EARTH,
        }
    };
}

/// Fill `p` with the defaults documented in the header — the patent's worked example.
///
/// # Safety
/// `p` is writable for one `sl_draim_params`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_draim_params_init(p: *mut SlDraimParams) {
    if p.is_null() {
        return;
    }
    // SAFETY: checked non-null just above.
    unsafe {
        *p = SlDraimParams {
            period_s: 172_800.0,
            eccentricity: 0.28,
            inclination_rad: DRAIM_INCLINATION_RAD,
            raan0_rad: 0.0,
            mu: MU_EARTH,
        }
    };
}

/// Generate a Flower constellation. Two-call protocol; see the header.
///
/// # Safety
/// `p` points at one initialised `sl_flower_params`; `out` is null or writable for `cap` states;
/// `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_flower(
    p: *const SlFlowerParams,
    out: *mut SlState,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if p.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_flower: null params pointer");
        }
        // SAFETY: checked non-null just above.
        let p = unsafe { *p };
        let flower = FlowerConstellation {
            petals: p.petals,
            days: p.days,
            satellites: p.satellites,
            eccentricity: p.eccentricity,
            inclination: Angle::from_radians(p.inclination_rad),
            arg_periapsis: Angle::from_radians(p.arg_periapsis_rad),
            raan: Angle::from_radians(p.raan_rad),
            mean_anomaly: Angle::from_radians(p.mean_anomaly_rad),
            frame_rate: p.frame_rate_rad_s,
            mu: p.mu,
        };
        let mut states = Vec::new();
        flower.generate(|e| {
            let (r, v) = e.to_state();
            states.push(from_state(r, v));
        });
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&states, out, cap, out_len) }
    })
}

/// Generate a 2-D Lattice Flower constellation. Two-call protocol; see the header.
///
/// # Safety
/// As [`sl_flower`], with `sl_lattice_params`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_lattice_flower(
    p: *const SlLatticeParams,
    out: *mut SlState,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if p.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_lattice_flower: null params pointer",
            );
        }
        // SAFETY: checked non-null just above.
        let p = unsafe { *p };
        let lattice = LatticeFlower {
            orbits: p.orbits,
            sats_per_orbit: p.sats_per_orbit,
            config: p.config,
            semi_major_axis: p.semi_major_axis_m,
            eccentricity: p.eccentricity,
            inclination: Angle::from_radians(p.inclination_rad),
            arg_periapsis: Angle::from_radians(p.arg_periapsis_rad),
            raan0: Angle::from_radians(p.raan0_rad),
            mu: p.mu,
        };
        let mut states = Vec::new();
        lattice.generate(|e| {
            let (r, v) = e.to_state();
            states.push(from_state(r, v));
        });
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&states, out, cap, out_len) }
    })
}

/// Build the Rust design from the C parameters.
fn draim_of(p: SlDraimParams) -> DraimTetrahedral {
    DraimTetrahedral::from_period(
        p.period_s,
        p.eccentricity,
        Angle::from_radians(p.inclination_rad),
        Angle::from_radians(p.raan0_rad),
        p.mu,
    )
}

/// Generate Draim's four satellites. Two-call protocol; see the header.
///
/// # Safety
/// As [`sl_flower`], with `sl_draim_params`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_draim(
    p: *const SlDraimParams,
    out: *mut SlState,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if p.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_draim: null params pointer");
        }
        // SAFETY: checked non-null just above.
        let mut states = Vec::new();
        draim_of(unsafe { *p }).generate(|e| {
            let (r, v) = e.to_state();
            states.push(from_state(r, v));
        });
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&states, out, cap, out_len) }
    })
}

/// `sl_draim_orbit`: the geometry of the orbit a Draim design flies, metres.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlDraimOrbit {
    /// Semi-major axis, metres.
    pub semi_major_axis_m: f64,
    /// Perigee radius from the centre of the Earth, metres.
    pub perigee_radius_m: f64,
    /// Apogee radius from the centre of the Earth, metres.
    pub apogee_radius_m: f64,
}

/// The Draim orbit's semi-major axis, perigee radius and apogee radius, metres.
///
/// # Safety
/// `p` points at one initialised `sl_draim_params`; `out` is writable for one `sl_draim_orbit`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_draim_geometry(p: *const SlDraimParams, out: *mut SlDraimOrbit) -> i32 {
    guard(|| {
        if p.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_draim_geometry: null pointer");
        }
        // SAFETY: checked non-null just above.
        let p = unsafe { *p };
        let a = draim_of(p).semi_major_axis;
        if !a.is_finite() {
            return fail(
                SL_ERR_DOMAIN,
                format!(
                    "sl_draim_geometry: no orbit for a period of {} s",
                    p.period_s
                ),
            );
        }
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe {
            *out = SlDraimOrbit {
                semi_major_axis_m: a,
                perigee_radius_m: a * (1.0 - p.eccentricity),
                apogee_radius_m: a * (1.0 + p.eccentricity),
            }
        };
        SL_OK
    })
}

/// Every position in `states` must be finite before it becomes an orbit: `KeplerianElements::from_state`
/// answers a NaN element set rather than refusing, and a NaN element set is a coverage answer of 0.
fn check_positions(who: &str, states: &[SlState]) -> Result<(), i32> {
    if let Some(i) = states.iter().position(|s| {
        !(s.position.x.is_finite() && s.position.y.is_finite() && s.position.z.is_finite())
    }) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: the position of state {i} is not finite"),
        ));
    }
    Ok(())
}

/// What the α-characteristic needs before it samples anything: satellites to sample, a gravitational
/// parameter to turn them into orbits, and three counts that are loop bounds.
fn check_alpha_inputs(
    who: &str,
    n: usize,
    mu: f64,
    fold: u32,
    grid_points: u32,
    time_samples: u32,
) -> Result<(), i32> {
    check_nonzero(who, "n", n)?;
    check_positive(who, "mu", mu, SL_ERR_INVALID_ARGUMENT)?;
    check_nonzero(who, "fold", fold)?;
    check_nonzero(who, "grid_points", grid_points)?;
    check_nonzero(who, "time_samples", time_samples)
}

/// The α-characteristic of a delta system: the smallest footprint half-angle at which the given
/// satellites continuously `fold`-fold cover the sphere, radians.
///
/// # Safety
/// `states` is readable for `n` `sl_state`; `out_rad` is writable for one `double`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_alpha_characteristic(
    states: *const SlState,
    n: usize,
    mu: f64,
    fold: u32,
    grid_points: u32,
    time_samples: u32,
    out_rad: *mut f64,
) -> i32 {
    guard(|| {
        if states.is_null() || out_rad.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_alpha_characteristic: null pointer",
            );
        }
        const WHO: &str = "sl_alpha_characteristic";
        if let Err(status) = check_alpha_inputs(WHO, n, mu, fold, grid_points, time_samples) {
            return status;
        }
        // SAFETY: the caller states `states` is readable for `n`.
        let states = unsafe { core::slice::from_raw_parts(states, n) };
        if let Err(status) = check_positions(WHO, states) {
            return status;
        }
        let mut elements = Vec::with_capacity(n);
        for s in states {
            let (r, v) = (to_vec3(s.position), to_vec3(s.velocity));
            elements.push(KeplerianElements::from_state(r, v, mu));
        }
        let alpha = constellation_alpha(&elements, fold, grid_points, time_samples);
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe { *out_rad = alpha };
        SL_OK
    })
}

/// The search's inputs: a footprint to clear, the orbit to fly it from, and four counts that bound
/// the search. The counts arrive as a named table because a refusal that says which one was zero is
/// the difference between a fixed call and a bisected one.
fn check_minimum_walker_inputs(
    who: &str,
    max_footprint_rad: f64,
    semi_major_axis_m: f64,
    mu: f64,
    counts: [(&str, u32); 4],
) -> Result<(), i32> {
    check_positive(
        who,
        "max_footprint_rad",
        max_footprint_rad,
        SL_ERR_INVALID_ARGUMENT,
    )?;
    check_positive(
        who,
        "semi_major_axis_m",
        semi_major_axis_m,
        SL_ERR_INVALID_ARGUMENT,
    )?;
    check_positive(who, "mu", mu, SL_ERR_INVALID_ARGUMENT)?;
    for (name, value) in counts {
        check_nonzero(who, name, value)?;
    }
    Ok(())
}

/// The fewest-satellite Walker Delta whose α-characteristic clears `max_footprint_rad`.
///
/// Writes the design into `out`, whose `pattern`, `inclination_rad`, `semi_major_axis_m` and `mu`
/// are the ones asked for and whose `planes`, `per_plane` and `phasing` are the answer — so the
/// result is directly a `sl_walker` input rather than three numbers to reassemble.
///
/// # Safety
/// `out` is writable for one `sl_walker_params`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_minimum_walker(
    max_footprint_rad: f64,
    inclination_rad: f64,
    semi_major_axis_m: f64,
    mu: f64,
    fold: u32,
    max_total: u32,
    grid_points: u32,
    time_samples: u32,
    out: *mut SlWalkerParams,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_minimum_walker: null out");
        }
        if let Err(status) = check_minimum_walker_inputs(
            "sl_minimum_walker",
            max_footprint_rad,
            semi_major_axis_m,
            mu,
            [
                ("fold", fold),
                ("max_total", max_total),
                ("grid_points", grid_points),
                ("time_samples", time_samples),
            ],
        ) {
            return status;
        }
        let Some(w) = constellation_minimum_walker(
            max_footprint_rad,
            Angle::from_radians(inclination_rad),
            semi_major_axis_m,
            mu,
            fold,
            max_total,
            grid_points,
            time_samples,
        ) else {
            return fail(
                SL_ERR_DOMAIN,
                format!(
                    "no Walker Delta of at most {max_total} satellites reaches a footprint of \
                     {max_footprint_rad} rad at this altitude"
                ),
            );
        };
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe {
            *out = SlWalkerParams {
                pattern: WALKER_DELTA,
                inclination_rad,
                planes: w.planes,
                per_plane: w.per_plane,
                phasing: w.phasing,
                semi_major_axis_m,
                mu,
            };
        }
        SL_OK
    })
}

// ── Coverage ──────────────────────────────────────────────────────────────────────────────────

/// `sl_sensor_kind`: a minimum elevation above the local horizon.
const SENSOR_ELEVATION_MASK: i32 = 0;
/// `sl_sensor_kind`: a side-looking SAR's incidence band.
const SENSOR_SIDE_LOOKING_SAR: i32 = 1;
/// `sl_sensor_kind`: an optical instrument slewable off nadir.
const SENSOR_AGILE_OPTICAL: i32 = 2;

/// `sl_sensor`: the instrument a coverage query is asked about.
///
/// One struct with a tag rather than three, because the two entry points that take it would
/// otherwise each need three overloads, and C has no way to spell one. Only the fields the tag
/// names are read; the rest may be anything, which is what `sl_sensor_init` leaves them as.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlSensor {
    /// Which of the three this is.
    pub kind: i32,
    /// `SL_SENSOR_ELEVATION_MASK`: the elevation a target must clear, radians.
    pub min_elevation_rad: f64,
    /// `SL_SENSOR_SIDE_LOOKING_SAR`: the near edge of the incidence band, radians.
    pub near_incidence_rad: f64,
    /// `SL_SENSOR_SIDE_LOOKING_SAR`: the far edge of the incidence band, radians.
    pub far_incidence_rad: f64,
    /// `SL_SENSOR_AGILE_OPTICAL`: how far off nadir the instrument slews, radians.
    pub max_off_nadir_rad: f64,
}

/// `sl_coverage_metrics`: what one ground cell saw.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlCoverageMetrics {
    /// Distinct passes over the cell.
    pub passes: u32,
    /// Fraction of the horizon the cell was covered for.
    pub coverage_fraction: f64,
    /// The longest gap between passes, seconds.
    pub max_gap_s: f64,
    /// The mean gap between passes, seconds.
    pub mean_gap_s: f64,
    /// The mean wait from an arbitrary instant to the next pass, seconds.
    pub mean_response_time_s: f64,
}

/// `sl_constellation_fom`: the whole-constellation figures of merit.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlConstellationFom {
    /// Ground cells the figures were computed over.
    pub point_count: u32,
    /// Mean coverage fraction.
    pub mean_coverage: f64,
    /// The worst cell's coverage fraction.
    pub min_coverage: f64,
    /// The worst cell's longest gap, seconds.
    pub worst_max_gap_s: f64,
    /// The mean of every cell's longest gap, seconds.
    pub mean_max_gap_s: f64,
    /// Mean response time, seconds.
    pub mean_response_time_s: f64,
    /// The worst cell's response time, seconds.
    pub worst_response_time_s: f64,
    /// Cells covered for the whole horizon.
    pub fully_covered_points: u32,
}

/// `sl_window`: one access interval, seconds from the start of the scan.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlWindow {
    /// When coverage opens.
    pub rise_s: f64,
    /// When coverage closes.
    pub set_s: f64,
}

/// Fill `s` with a sensor of `kind` and nothing else set.
///
/// # Safety
/// `s` is writable for one `sl_sensor`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sensor_init(s: *mut SlSensor, kind: i32) {
    if s.is_null() {
        return;
    }
    // SAFETY: checked non-null just above.
    unsafe {
        *s = SlSensor {
            kind,
            min_elevation_rad: 0.0,
            near_incidence_rad: 0.0,
            far_incidence_rad: 0.0,
            max_off_nadir_rad: 0.0,
        }
    };
}

/// Which of the two published fits a conversion is asking for. Named rather than a `bool`, because
/// `true` at a call site says nothing about which altitude it is asking for.
enum RefractionDirection {
    /// Saemundsson (1986), true to apparent.
    ToApparent,
    /// Bennett (1982), apparent to true.
    ToTrue,
}

/// Both refraction conversions, as one safe function, because the two C entry points differ only in
/// which fit they call and which altitude they name.
///
/// Safe on purpose: `cargo mutants` 27.1.0 generates nothing inside an `unsafe fn` body, so a
/// predicate written there is unmeasured. The pointer work stays in the entry point and the
/// arithmetic and the domain check stay here (#260).
fn refraction_conversion(
    who: &str,
    altitude_rad: f64,
    pressure_hpa: f64,
    temperature_k: f64,
    direction: RefractionDirection,
) -> Result<f64, i32> {
    check_finite(who, "altitude_rad", altitude_rad)?;
    check_positive(who, "pressure_hpa", pressure_hpa, SL_ERR_DOMAIN)?;
    check_positive(who, "temperature_k", temperature_k, SL_ERR_DOMAIN)?;
    let refraction = Refraction {
        pressure_hpa,
        temperature_k,
    };
    let bent = match direction {
        RefractionDirection::ToApparent => {
            refraction.apparent(TrueAltitude(altitude_rad)).map(|h| h.0)
        }
        RefractionDirection::ToTrue => refraction
            .geometric(ApparentAltitude(altitude_rad))
            .map(|h| h.0),
    };
    bent.ok_or_else(|| {
        fail(
            SL_ERR_DOMAIN,
            format!(
                "{who}: {altitude_rad} rad is outside the fit's domain -- Saemundsson is defined \
                 from -1 degree to the zenith and Bennett from the horizon to the zenith"
            ),
        )
    })
}

/// True to apparent altitude, Saemundsson (1986).
///
/// # Safety
/// `out_apparent_rad` is writable for one `double`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_refraction_apparent(
    true_altitude_rad: f64,
    pressure_hpa: f64,
    temperature_k: f64,
    out_apparent_rad: *mut f64,
) -> i32 {
    guard(|| {
        if out_apparent_rad.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_refraction_apparent: null pointer",
            );
        }
        match refraction_conversion(
            "sl_refraction_apparent",
            true_altitude_rad,
            pressure_hpa,
            temperature_k,
            RefractionDirection::ToApparent,
        ) {
            // SAFETY: checked non-null; written only now that the answer exists.
            Ok(h) => {
                unsafe { *out_apparent_rad = h };
                SL_OK
            }
            Err(status) => status,
        }
    })
}

/// Apparent to true altitude, Bennett (1982) — the conversion an `SL_SENSOR_ELEVATION_MASK` needs.
///
/// # Safety
/// `out_true_rad` is writable for one `double`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_refraction_geometric(
    apparent_altitude_rad: f64,
    pressure_hpa: f64,
    temperature_k: f64,
    out_true_rad: *mut f64,
) -> i32 {
    guard(|| {
        if out_true_rad.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_refraction_geometric: null pointer",
            );
        }
        match refraction_conversion(
            "sl_refraction_geometric",
            apparent_altitude_rad,
            pressure_hpa,
            temperature_k,
            RefractionDirection::ToTrue,
        ) {
            // SAFETY: checked non-null; written only now that the answer exists.
            Ok(h) => {
                unsafe { *out_true_rad = h };
                SL_OK
            }
            Err(status) => status,
        }
    })
}

/// Which way a direction is being carried across the aberration map. Named rather than a `bool`, for
/// the reason `RefractionDirection` is: at the call site `true` says nothing about which direction
/// the caller holds and which one they want.
enum AberrationDirection {
    /// Natural to apparent — predicting what a moving observer will see.
    ToApparent,
    /// Apparent to natural — reducing what one did see.
    ToNatural,
}

/// Both aberration maps, as one safe function, for the reason `refraction_conversion` is one: the
/// two entry points differ only in which way they carry the direction.
///
/// Safe on purpose (#260): `cargo mutants` 27.1.0 generates nothing inside an `unsafe fn` body, so
/// the two domain checks below would be unmeasured if they lived in the entry point.
fn aberrate(
    who: &str,
    velocity: Vec3,
    direction: Vec3,
    direction_name: &str,
    which: AberrationDirection,
) -> Result<Vec3, i32> {
    check_all_finite(who, "velocity_mps", &[velocity.x, velocity.y, velocity.z])?;
    check_all_finite(
        who,
        direction_name,
        &[direction.x, direction.y, direction.z],
    )?;
    let aberration = Aberration::new(velocity).ok_or_else(|| {
        fail(
            SL_ERR_DOMAIN,
            format!(
                "{who}: |velocity_mps| is {} m/s, at or above the speed of light",
                velocity.norm()
            ),
        )
    })?;
    let carried = match which {
        AberrationDirection::ToApparent => aberration.apply(direction),
        AberrationDirection::ToNatural => aberration.remove(direction),
    };
    carried.ok_or_else(|| {
        fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: {direction_name} has zero length, so it is not a direction"),
        )
    })
}

/// The velocity a station is carried at by the rotation of the body it stands on. See the header.
///
/// # Safety
/// `station_m` and `angular_velocity_rad_s` point at initialised `sl_vec3`; `out_velocity_mps` is
/// writable for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_diurnal_velocity(
    station_m: *const SlVec3,
    angular_velocity_rad_s: *const SlVec3,
    out_velocity_mps: *mut SlVec3,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_diurnal_velocity";
        if station_m.is_null() || angular_velocity_rad_s.is_null() || out_velocity_mps.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: all three checked non-null just above.
        let (r, w) = unsafe { (to_vec3(*station_m), to_vec3(*angular_velocity_rad_s)) };
        if let Err(status) = check_all_finite(WHO, "station_m", &[r.x, r.y, r.z]) {
            return status;
        }
        if let Err(status) = check_all_finite(WHO, "angular_velocity_rad_s", &[w.x, w.y, w.z]) {
            return status;
        }
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe { *out_velocity_mps = from_vec3(w.cross(r)) };
        SL_OK
    })
}

/// Natural direction to apparent direction. See the header.
///
/// # Safety
/// `velocity_mps` and `natural` point at initialised `sl_vec3`; `out_apparent` is writable for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_aberration_apply(
    velocity_mps: *const SlVec3,
    natural: *const SlVec3,
    out_apparent: *mut SlVec3,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_aberration_apply";
        if velocity_mps.is_null() || natural.is_null() || out_apparent.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: both checked non-null just above.
        let (v, u) = unsafe { (to_vec3(*velocity_mps), to_vec3(*natural)) };
        match aberrate(WHO, v, u, "natural", AberrationDirection::ToApparent) {
            // SAFETY: checked non-null; written only now that the answer exists.
            Ok(apparent) => {
                unsafe { *out_apparent = from_vec3(apparent) };
                SL_OK
            }
            Err(status) => status,
        }
    })
}

/// Apparent direction to natural direction. See the header.
///
/// # Safety
/// `velocity_mps` and `apparent` point at initialised `sl_vec3`; `out_natural` is writable for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_aberration_remove(
    velocity_mps: *const SlVec3,
    apparent: *const SlVec3,
    out_natural: *mut SlVec3,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_aberration_remove";
        if velocity_mps.is_null() || apparent.is_null() || out_natural.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: both checked non-null just above.
        let (v, u) = unsafe { (to_vec3(*velocity_mps), to_vec3(*apparent)) };
        match aberrate(WHO, v, u, "apparent", AberrationDirection::ToNatural) {
            // SAFETY: checked non-null; written only now that the answer exists.
            Ok(natural) => {
                unsafe { *out_natural = from_vec3(natural) };
                SL_OK
            }
            Err(status) => status,
        }
    })
}

/// The three sensors, as one type the two coverage entry points can dispatch on.
enum Instrument {
    Elevation(ElevationMask),
    Sar(SideLookingSAR),
    Agile(AgileOptical),
}

/// Read a `sl_sensor`, or name what is wrong with it. The tag is checked before the fields, so an
/// out-of-range tag never reads a value that was never set.
fn instrument_of(who: &str, s: SlSensor) -> Result<Instrument, i32> {
    let built = match s.kind {
        SENSOR_ELEVATION_MASK => ElevationMask::new(s.min_elevation_rad).map(Instrument::Elevation),
        SENSOR_SIDE_LOOKING_SAR => {
            SideLookingSAR::new(s.near_incidence_rad, s.far_incidence_rad).map(Instrument::Sar)
        }
        SENSOR_AGILE_OPTICAL => AgileOptical::new(s.max_off_nadir_rad).map(Instrument::Agile),
        other => {
            return Err(fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{who}: sensor tag {other} is not an sl_sensor_kind"),
            ));
        }
    };
    built.map_err(|e| fail(SL_ERR_DOMAIN, format!("{who}: {e:?}")))
}

/// The circular-orbit ground track the coverage engine scans, ECEF.
///
/// Analytic rather than integrated, and identical to the Python bridge's, so the two answer the
/// same question: a coverage sweep asks about geometry over days, where a J2 secular drift changes
/// the answer far less than the sensor model does, and an integration would put a step-size choice
/// between the caller and a figure of merit.
fn circular_ground_track(a: f64, inc: f64, raan: f64, phase: f64) -> impl Fn(f64) -> Vec3 {
    let n = (MU_EARTH / (a * a * a)).sqrt();
    move |t: f64| {
        let u = phase + n * t;
        let (cu, su) = (u.cos(), u.sin());
        let (cr, sr) = (raan.cos(), raan.sin());
        let (ci, si) = (inc.cos(), inc.sin());
        let eci = Vec3::new(
            a * (cu * cr - su * sr * ci),
            a * (cu * sr + su * cr * ci),
            a * (su * si),
        );
        Mat3::rot3(spacelab_frames::EARTH_ROTATION_RATE * t) * eci
    }
}

/// The scalars every coverage query shares, checked once.
fn check_coverage_orbit(
    who: &str,
    a: f64,
    inc: f64,
    raan: f64,
    phase: f64,
    horizon: f64,
    tol: f64,
) -> Result<(), i32> {
    if !(a.is_finite() && a > 0.0) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: the semi-major axis must be finite and positive"),
        ));
    }
    if !(inc.is_finite() && raan.is_finite() && phase.is_finite()) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: the orbit angles must be finite"),
        ));
    }
    if !(horizon.is_finite() && horizon > 0.0) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: horizon must be finite and positive"),
        ));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: tol must be finite and positive"),
        ));
    }
    Ok(())
}

/// Ground cells on a spherical Earth from a latitude and a longitude grid, both radians.
///
/// Writes `lat_count * lon_count` cells, longitude-major within each latitude, as
/// `(centre_ecef_m, zenith_unit)` pairs in the position and velocity slots of an `sl_state` — the
/// shape the two coverage entry points take. Radians here, where the Python binding takes degrees:
/// the rest of this API is radians and a C caller has no keyword to remind them which.
///
/// # Safety
/// `lats_rad` and `lons_rad` are readable for their counts; `out` is null or writable for
/// `lat_count * lon_count` states; `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_grid_cells(
    lats_rad: *const f64,
    lat_count: usize,
    lons_rad: *const f64,
    lon_count: usize,
    out: *mut SlState,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if lats_rad.is_null() || lons_rad.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_grid_cells: null pointer");
        }
        const WHO: &str = "sl_grid_cells";
        if let Err(status) = check_nonzero(WHO, "lat_count", lat_count)
            .and_then(|()| check_nonzero(WHO, "lon_count", lon_count))
        {
            return status;
        }
        // SAFETY: the caller states each array is readable for its count.
        let (lats, lons) = unsafe {
            (
                core::slice::from_raw_parts(lats_rad, lat_count),
                core::slice::from_raw_parts(lons_rad, lon_count),
            )
        };
        if let Err(status) = check_all_finite(WHO, "lats_rad", lats)
            .and_then(|()| check_all_finite(WHO, "lons_rad", lons))
        {
            return status;
        }
        let mut cells = Vec::with_capacity(lat_count * lon_count);
        for &phi in lats {
            for &lam in lons {
                let up = Vec3::new(phi.cos() * lam.cos(), phi.cos() * lam.sin(), phi.sin());
                cells.push(from_state(up.scale(RE_EARTH), up));
            }
        }
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&cells, out, cap, out_len) }
    })
}

/// Per-cell coverage and revisit metrics for one circular orbit over `[0, horizon]` seconds.
///
/// # Safety
/// `sensor` points at one initialised `sl_sensor`; `cells` is readable for `n` and `out` writable
/// for `n` `sl_coverage_metrics`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_coverage_scan(
    a_m: f64,
    inclination_rad: f64,
    raan_rad: f64,
    phase_rad: f64,
    sensor: *const SlSensor,
    cells: *const SlState,
    n: usize,
    horizon_s: f64,
    tol_s: f64,
    threads: u32,
    out: *mut SlCoverageMetrics,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_coverage_scan";
        if sensor.is_null() || cells.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_nonzero(WHO, "n", n) {
            return status;
        }
        if let Err(status) = check_coverage_orbit(
            WHO,
            a_m,
            inclination_rad,
            raan_rad,
            phase_rad,
            horizon_s,
            tol_s,
        ) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let instrument = match instrument_of(WHO, unsafe { *sensor }) {
            Ok(instrument) => instrument,
            Err(status) => return status,
        };
        // SAFETY: the caller states `cells` is readable for `n`.
        let cells = unsafe { core::slice::from_raw_parts(cells, n) };
        let cells: Vec<(Vec3, Vec3)> = cells
            .iter()
            .map(|c| (to_vec3(c.position), to_vec3(c.velocity)))
            .collect();
        let sat_at = circular_ground_track(a_m, inclination_rad, raan_rad, phase_rad);
        let orbit = OrbitBound {
            semi_major_axis: a_m,
            eccentricity: 0.0,
            mu: MU_EARTH,
        };
        let rate = spacelab_frames::EARTH_ROTATION_RATE;
        // Zero threads means "as many as you like", which is the one answer a caller who does not
        // want to think about it can give.
        let threads = if threads == 0 { 1 } else { threads as usize };
        let metrics = match &instrument {
            Instrument::Elevation(s) => revisit_map(
                &cells, s, &orbit, rate, sat_at, 0.0, horizon_s, tol_s, threads,
            ),
            Instrument::Sar(s) => revisit_map(
                &cells, s, &orbit, rate, sat_at, 0.0, horizon_s, tol_s, threads,
            ),
            Instrument::Agile(s) => revisit_map(
                &cells, s, &orbit, rate, sat_at, 0.0, horizon_s, tol_s, threads,
            ),
        };
        for (i, m) in metrics.iter().enumerate() {
            // SAFETY: the caller states `out` is writable for `n`, and `metrics.len() == n`.
            unsafe {
                *out.add(i) = SlCoverageMetrics {
                    passes: m.passes,
                    coverage_fraction: m.coverage_fraction,
                    max_gap_s: m.max_gap,
                    mean_gap_s: m.mean_gap,
                    mean_response_time_s: m.mean_response_time,
                };
            }
        }
        SL_OK
    })
}

/// The raw access windows of one circular orbit over a single ground cell. Two-call protocol.
///
/// # Safety
/// `sensor` and `cell` each point at one initialised value; `out` is null or writable for `cap`
/// `sl_window`; `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_access_windows(
    a_m: f64,
    inclination_rad: f64,
    raan_rad: f64,
    phase_rad: f64,
    sensor: *const SlSensor,
    cell: *const SlState,
    horizon_s: f64,
    tol_s: f64,
    out: *mut SlWindow,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_access_windows";
        if sensor.is_null() || cell.is_null() || out_len.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_coverage_orbit(
            WHO,
            a_m,
            inclination_rad,
            raan_rad,
            phase_rad,
            horizon_s,
            tol_s,
        ) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let instrument = match instrument_of(WHO, unsafe { *sensor }) {
            Ok(instrument) => instrument,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states it is initialised.
        let cell = unsafe { *cell };
        let (centre, zenith) = (to_vec3(cell.position), to_vec3(cell.velocity));
        let sat_at = circular_ground_track(a_m, inclination_rad, raan_rad, phase_rad);
        let orbit = OrbitBound {
            semi_major_axis: a_m,
            eccentricity: 0.0,
            mu: MU_EARTH,
        };
        let rate = spacelab_frames::EARTH_ROTATION_RATE;
        let mut windows = Vec::new();
        let mut emit = |w: Window| {
            windows.push(SlWindow {
                rise_s: w.rise,
                set_s: w.set,
            });
        };
        match &instrument {
            Instrument::Elevation(s) => coverage_access_windows(
                s, &orbit, rate, centre, zenith, sat_at, 0.0, horizon_s, tol_s, &mut emit,
            ),
            Instrument::Sar(s) => coverage_access_windows(
                s, &orbit, rate, centre, zenith, sat_at, 0.0, horizon_s, tol_s, &mut emit,
            ),
            Instrument::Agile(s) => coverage_access_windows(
                s, &orbit, rate, centre, zenith, sat_at, 0.0, horizon_s, tol_s, &mut emit,
            ),
        }
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&windows, out, cap, out_len) }
    })
}

/// Aggregate per-cell metrics into whole-constellation figures of merit.
///
/// `weights` is null for equal weighting, or `n` values — `cos(latitude)` is the usual choice, so
/// that a lat/lon grid's crowded poles do not outvote its equator.
///
/// # Safety
/// `cells` is readable for `n`; `weights` is null or readable for `n`; `out` is writable for one
/// `sl_constellation_fom`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_figures_of_merit(
    cells: *const SlCoverageMetrics,
    n: usize,
    weights: *const f64,
    out: *mut SlConstellationFom,
) -> i32 {
    guard(|| {
        if cells.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_figures_of_merit: null pointer");
        }
        const WHO: &str = "sl_figures_of_merit";
        if let Err(status) = check_nonzero(WHO, "n", n) {
            return status;
        }
        // SAFETY: the caller states `cells` is readable for `n`.
        let cells = unsafe { core::slice::from_raw_parts(cells, n) };
        // SAFETY: the caller states `weights` is null or readable for `n`.
        let weights =
            (!weights.is_null()).then(|| unsafe { core::slice::from_raw_parts(weights, n) });
        if let Err(status) = check_all_finite(WHO, "weights", weights.unwrap_or(&[])) {
            return status;
        }
        // Only the five fields the C struct carries are read by `constellation_fom`; the rest are
        // per-pass durations the metrics entry point does not report, and inventing values for
        // them here would put numbers in an answer that nothing measured.
        let metrics: Vec<CoverageMetrics> = cells
            .iter()
            .map(|c| CoverageMetrics {
                passes: c.passes,
                covered_seconds: 0.0,
                coverage_fraction: c.coverage_fraction,
                max_gap: c.max_gap_s,
                mean_gap: c.mean_gap_s,
                mean_response_time: c.mean_response_time_s,
                longest_pass: 0.0,
                shortest_pass: 0.0,
            })
            .collect();
        let f = coverage_constellation_fom(&metrics, weights);
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe {
            *out = SlConstellationFom {
                point_count: f.point_count,
                mean_coverage: f.mean_coverage,
                min_coverage: f.min_coverage,
                worst_max_gap_s: f.worst_max_gap,
                mean_max_gap_s: f.mean_max_gap,
                mean_response_time_s: f.mean_response_time,
                worst_response_time_s: f.worst_response_time,
                fully_covered_points: f.fully_covered_points,
            };
        }
        SL_OK
    })
}

// ── GNSS ──────────────────────────────────────────────────────────────────────────────────────

/// `sl_gnss_system`: which constellation a broadcast record belongs to.
const GNSS_GPS: i32 = 0;
const GNSS_GALILEO: i32 = 1;
const GNSS_BEIDOU: i32 = 2;
const GNSS_QZSS: i32 = 3;
const GNSS_NAVIC: i32 = 4;

/// `sl_gnss_state`: one satellite's broadcast position and velocity, with what identifies it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlGnssState {
    /// Which constellation, an `sl_gnss_system`.
    pub system: i32,
    /// The PRN within that constellation. A PRN alone is ambiguous in a mixed file.
    pub prn: u32,
    /// ECEF position and velocity, metres and metres per second.
    pub state: SlState,
}

/// `sl_glonass_state`: one GLONASS satellite, which is numbered by slot rather than by PRN.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlGlonassState {
    /// The orbital slot.
    pub slot: u32,
    /// PZ-90 position and velocity, metres and metres per second.
    pub state: SlState,
}

/// `sl_dop`: the five dilution-of-precision figures.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlDop {
    /// Geometric DOP — position and clock together.
    pub gdop: f64,
    /// Position DOP.
    pub pdop: f64,
    /// Horizontal DOP.
    pub hdop: f64,
    /// Vertical DOP.
    pub vdop: f64,
    /// Time DOP.
    pub tdop: f64,
}

/// `sl_mapping`: a tropospheric mapping function's two factors.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlMapping {
    /// The hydrostatic (dry) mapping factor.
    pub hydrostatic: f64,
    /// The wet mapping factor.
    pub wet: f64,
}

/// The GNSS crate speaks in flat `[f64; 3]` triples rather than in `Vec3`.
fn state_of(position: [f64; 3], velocity: [f64; 3]) -> SlState {
    SlState {
        position: SlVec3 {
            x: position[0],
            y: position[1],
            z: position[2],
        },
        velocity: SlVec3 {
            x: velocity[0],
            y: velocity[1],
            z: velocity[2],
        },
    }
}

/// Borrow a length-counted string. RINEX text is a file, not a token, so it is passed with its
/// length rather than NUL-terminated: `MAX_C_STRING` bounds the scan `sl_sgp4_open` does, and a
/// navigation file is far longer than that bound.
///
/// # Safety
/// `ptr` is null, or readable for `len` bytes.
unsafe fn borrow_sized_str<'a>(ptr: *const c_char, len: usize) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the caller states `ptr` is readable for `len` bytes.
    let bytes = unsafe { core::slice::from_raw_parts(ptr.cast::<u8>(), len) };
    core::str::from_utf8(bytes).ok()
}

/// Propagate every GPS/Galileo/BeiDou/QZSS/NavIC record in a RINEX-3 navigation file. Two-call
/// protocol; see the header.
///
/// # Safety
/// `text` is readable for `text_len` bytes; `out` is null or writable for `cap` `sl_gnss_state`;
/// `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_rinex_gnss_positions(
    text: *const c_char,
    text_len: usize,
    tk_seconds: f64,
    out: *mut SlGnssState,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: the caller states `text` is readable for `text_len`.
        let Some(text) = (unsafe { borrow_sized_str(text, text_len) }) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_rinex_gnss_positions: text is null or not UTF-8",
            );
        };
        if let Err(status) = check_finite("sl_rinex_gnss_positions", "tk_seconds", tk_seconds) {
            return status;
        }
        let records: Vec<SlGnssState> = parse_nav(text)
            .into_iter()
            .map(|e| {
                let (r, v) = e.position_velocity_ecef(tk_seconds);
                SlGnssState {
                    system: match e.system {
                        GnssSystem::Gps => GNSS_GPS,
                        GnssSystem::Galileo => GNSS_GALILEO,
                        GnssSystem::Beidou => GNSS_BEIDOU,
                        GnssSystem::Qzss => GNSS_QZSS,
                        GnssSystem::Navic => GNSS_NAVIC,
                    },
                    prn: u32::from(e.prn),
                    state: state_of(r, v),
                }
            })
            .collect();
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&records, out, cap, out_len) }
    })
}

/// Propagate every GLONASS record in a RINEX-3 navigation file. Two-call protocol.
///
/// # Safety
/// `text` is readable for `text_len` bytes; `out` is null or writable for `cap`
/// `sl_glonass_state`; `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_rinex_glonass_positions(
    text: *const c_char,
    text_len: usize,
    dt_seconds: f64,
    out: *mut SlGlonassState,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: the caller states `text` is readable for `text_len`.
        let Some(text) = (unsafe { borrow_sized_str(text, text_len) }) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_rinex_glonass_positions: text is null or not UTF-8",
            );
        };
        if let Err(status) = check_finite("sl_rinex_glonass_positions", "dt_seconds", dt_seconds) {
            return status;
        }
        let records: Vec<SlGlonassState> = parse_glonass_nav(text)
            .into_iter()
            .map(|g| {
                let (r, v) = g.position_velocity_pz90(dt_seconds);
                SlGlonassState {
                    slot: u32::from(g.slot),
                    state: state_of(r, v),
                }
            })
            .collect();
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&records, out, cap, out_len) }
    })
}

// ── RINEX 3 observation files ─────────────────────────────────────────────────────────────────
//
// The reader #312 added, as three flat entry points rather than one that hands the parsed file
// over. `RinexObs` is a per-constellation descriptor map over epochs of per-satellite *ragged*
// slot lists, and nothing ragged crosses this boundary without an opaque handle of its own. What
// a caller opens an observation file for is the header, the descriptors one constellation
// declares, and one satellite's series of one descriptor — so those are what it gets, each
// re-parsing the text the way `sl_rinex_gnss_positions` does.

/// Bytes `sl_obs_header::marker_name` holds: the sixty columns `MARKER NAME` occupies, and the
/// terminator.
const MARKER_NAME_BYTES: usize = 61;

/// Bytes `sl_obs_descriptor::code` holds: the three columns a RINEX 3 descriptor occupies, and the
/// terminator.
const OBS_CODE_BYTES: usize = 4;

/// `sl_time_system`, as the library names the scales.
const TIME_SYSTEM_GPS: i32 = 0;
const TIME_SYSTEM_GLO: i32 = 1;
const TIME_SYSTEM_GAL: i32 = 2;
const TIME_SYSTEM_TAI: i32 = 3;
const TIME_SYSTEM_UTC: i32 = 4;
const TIME_SYSTEM_QZS: i32 = 5;
const TIME_SYSTEM_BDT: i32 = 6;
const TIME_SYSTEM_IRN: i32 = 7;

/// `sl_obs_header`: the header records this reader interprets.
///
/// The three optional fields carry NaN where the header declares none, which is this ABI's
/// established way of saying absent — `sl_decay_params_init` leaves its unguessable fields the
/// same way.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlObsHeader {
    /// Format version from `RINEX VERSION / TYPE`.
    pub version: f64,
    /// `INTERVAL`, seconds, or NaN where the header declares none.
    pub interval: f64,
    /// `APPROX POSITION XYZ`, metres in the file's terrestrial frame, or all-NaN where absent.
    pub approx_position: [f64; 3],
    /// `ANTENNA: DELTA H/E/N`, metres — height above the marker, then east and north eccentricity
    /// — or all-NaN where absent. Not a Cartesian triple, which is why it is not an `sl_vec3`.
    pub antenna_delta_hen: [f64; 3],
    /// The scale `TIME OF FIRST OBS` names, an `sl_time_system`.
    pub time_system: i32,
    /// Satellite-system character of the file, or `M` for mixed.
    pub file_system: c_char,
    /// `MARKER NAME`, NUL-terminated, empty where the record is absent.
    pub marker_name: [c_char; MARKER_NAME_BYTES],
}

/// `sl_obs_descriptor`: one declared observation descriptor, e.g. `C1C`, NUL-terminated.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlObsDescriptor {
    /// The descriptor as the header writes it, with the blank third column of `I` and `X` dropped.
    pub code: [c_char; OBS_CODE_BYTES],
}

/// `sl_obs_sample`: one satellite's observation of one descriptor, at one epoch.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlObsSample {
    /// The instant, seconds from J2000 in the file's own time system.
    pub seconds_j2000: f64,
    /// The value, in the unit its descriptor's kind carries — metres for `C`, **cycles** for `L`.
    pub value: f64,
    /// The receiver's own loss-of-lock bit: 1 where it declares a slip, 0 otherwise.
    pub lost_lock: i32,
}

/// The scale as the header names it.
fn time_system_code(system: TimeSystem) -> i32 {
    match system {
        TimeSystem::Gps => TIME_SYSTEM_GPS,
        TimeSystem::Glo => TIME_SYSTEM_GLO,
        TimeSystem::Gal => TIME_SYSTEM_GAL,
        TimeSystem::Tai => TIME_SYSTEM_TAI,
        TimeSystem::Utc => TIME_SYSTEM_UTC,
        TimeSystem::Qzs => TIME_SYSTEM_QZS,
        TimeSystem::Bdt => TIME_SYSTEM_BDT,
        TimeSystem::Irn => TIME_SYSTEM_IRN,
    }
}

/// One ASCII column as its single-byte C field holds it, or 0 where the file put something wider
/// there: a `char` is up to four bytes and the field is one, so the alternative to refusing is a
/// truncation that reads as a different constellation.
fn ascii_column(c: char) -> c_char {
    #[expect(
        clippy::cast_possible_wrap,
        reason = "the cast is reached only for a byte below 0x80"
    )]
    match u8::try_from(c) {
        Ok(byte) if byte.is_ascii() => byte as c_char,
        _ => 0,
    }
}

/// Read an observation file, or a refusal naming what the reader made of the text.
fn parse_obs(who: &str, text: &str) -> Result<RinexObs, i32> {
    RinexObs::parse(text).map_err(|e| fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: {e}")))
}

/// The constellation letter a caller named, or a refusal.
fn obs_system(who: &str, system: c_char) -> Result<char, i32> {
    match u8::try_from(system).ok().map(char::from) {
        Some(letter) if letter.is_ascii_alphabetic() => Ok(letter),
        _ => Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: a constellation is a letter, not the byte {system}"),
        )),
    }
}

/// The header as its C mirror, absent fields as NaN.
fn obs_header_of(header: &ObsHeader) -> SlObsHeader {
    let absent = [f64::NAN; 3];
    let mut out = SlObsHeader {
        version: header.version,
        interval: header.interval.unwrap_or(f64::NAN),
        approx_position: header.approx_position.unwrap_or(absent),
        antenna_delta_hen: header.antenna_delta_hen.unwrap_or(absent),
        time_system: time_system_code(header.time_system),
        file_system: ascii_column(header.file_system),
        marker_name: [0; MARKER_NAME_BYTES],
    };
    copy_to_c_buffer(
        &header.marker_name,
        out.marker_name.as_mut_ptr(),
        MARKER_NAME_BYTES,
    );
    out
}

/// The descriptor as C reads it: the written form with the blank third column trimmed, which is
/// also the form `sl_rinex_obs_series` matches on.
fn obs_descriptor_of(code: &ObsCode) -> SlObsDescriptor {
    let mut out = SlObsDescriptor {
        code: [0; OBS_CODE_BYTES],
    };
    copy_to_c_buffer(
        code.to_string().trim_end(),
        out.code.as_mut_ptr(),
        OBS_CODE_BYTES,
    );
    out
}

/// The descriptors `system` declares, in the order every data record for it reports them. A
/// constellation the file never mentions declares none, which is zero descriptors and not an error
/// — the same reading `sl_rinex_gnss_positions` gives a GPS-only file asked for GLONASS.
fn obs_descriptors_of(header: &ObsHeader, system: char) -> Vec<SlObsDescriptor> {
    header
        .obs_types
        .get(&system)
        .map(|codes| codes.iter().map(obs_descriptor_of).collect())
        .unwrap_or_default()
}

/// One satellite's series of one descriptor, skipping the epochs it is absent from or carries no
/// value at.
///
/// A descriptor the header never declared for that constellation is a refusal rather than an empty
/// series, which is the difference between "not observed" and "asked for the wrong thing".
fn obs_series_of(
    who: &str,
    file: &RinexObs,
    system: char,
    prn: u8,
    code: &str,
) -> Result<Vec<SlObsSample>, i32> {
    let slot = file
        .header
        .obs_types
        .get(&system)
        .and_then(|codes| codes.iter().position(|c| c.to_string().trim_end() == code))
        .ok_or_else(|| {
            fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{who}: the header declares no {code:?} for constellation {system:?}"),
            )
        })?;
    Ok(file
        .epochs
        .iter()
        .filter_map(|epoch| {
            let record = epoch
                .satellites
                .iter()
                .find(|s| s.sat.system == system && s.sat.prn == prn)?;
            let observation = (*record.values.get(slot)?)?;
            Some(SlObsSample {
                seconds_j2000: epoch.seconds_j2000,
                value: observation.value,
                lost_lock: i32::from(observation.lost_lock()),
            })
        })
        .collect())
}

/// The header records of a RINEX 3 observation file. See the header.
///
/// # Safety
/// `text` is readable for `text_len` bytes; `out` is writable for one `sl_obs_header`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_rinex_obs_header(
    text: *const c_char,
    text_len: usize,
    out: *mut SlObsHeader,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_rinex_obs_header";
        // SAFETY: the caller states `text` is readable for `text_len`.
        let Some(text) = (unsafe { borrow_sized_str(text, text_len) }) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: text is null or not UTF-8"),
            );
        };
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out"));
        }
        let file = match parse_obs(WHO, text) {
            Ok(file) => file,
            Err(status) => return status,
        };
        // SAFETY: `out` was checked non-null above.
        unsafe { *out = obs_header_of(&file.header) };
        SL_OK
    })
}

/// The observation descriptors one constellation declares. Two-call protocol; see the header.
///
/// # Safety
/// `text` is readable for `text_len` bytes; `out` is null or writable for `cap`
/// `sl_obs_descriptor`; `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_rinex_obs_descriptors(
    text: *const c_char,
    text_len: usize,
    system: c_char,
    out: *mut SlObsDescriptor,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_rinex_obs_descriptors";
        // SAFETY: the caller states `text` is readable for `text_len`.
        let Some(text) = (unsafe { borrow_sized_str(text, text_len) }) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: text is null or not UTF-8"),
            );
        };
        let letter = match obs_system(WHO, system) {
            Ok(letter) => letter,
            Err(status) => return status,
        };
        let file = match parse_obs(WHO, text) {
            Ok(file) => file,
            Err(status) => return status,
        };
        let codes = obs_descriptors_of(&file.header, letter);
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&codes, out, cap, out_len) }
    })
}

/// One satellite's series of one observation descriptor. Two-call protocol; see the header.
///
/// # Safety
/// `text` is readable for `text_len` bytes; `code` is a NUL-terminated string; `out` is null or
/// writable for `cap` `sl_obs_sample`; `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_rinex_obs_series(
    text: *const c_char,
    text_len: usize,
    system: c_char,
    prn: u32,
    code: *const c_char,
    out: *mut SlObsSample,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_rinex_obs_series";
        // SAFETY: the caller states `text` is readable for `text_len`.
        let Some(text) = (unsafe { borrow_sized_str(text, text_len) }) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: text is null or not UTF-8"),
            );
        };
        // SAFETY: the caller states `code` is NUL-terminated; a descriptor is a token, not a file.
        let Some(code) = (unsafe { borrow_c_str(code) }) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: code is null, unterminated or not UTF-8"),
            );
        };
        let letter = match obs_system(WHO, system) {
            Ok(letter) => letter,
            Err(status) => return status,
        };
        let Ok(prn) = u8::try_from(prn) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: {prn} is not a PRN the format can write"),
            );
        };
        let file = match parse_obs(WHO, text) {
            Ok(file) => file,
            Err(status) => return status,
        };
        let samples = match obs_series_of(WHO, &file, letter, prn, code) {
            Ok(samples) => samples,
            Err(status) => return status,
        };
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&samples, out, cap, out_len) }
    })
}

/// The GLONASS FDMA L1 carrier frequency for channel `k`, hertz. Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_glonass_l1_frequency(channel: i32) -> f64 {
    match i8::try_from(channel) {
        Ok(k) => gnss_glonass_l1(k),
        Err(_) => f64::NAN,
    }
}

/// The GLONASS FDMA L2 carrier frequency for channel `k`, hertz. Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_glonass_l2_frequency(channel: i32) -> f64 {
    match i8::try_from(channel) {
        Ok(k) => gnss_glonass_l2(k),
        Err(_) => f64::NAN,
    }
}

/// Dilution of precision for a receiver seeing `n` satellites, all ECEF metres.
///
/// # Safety
/// `receiver` points at one `sl_vec3`; `satellites` is readable for `n`; `out` is writable for one
/// `sl_dop`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_dilution_of_precision(
    receiver: *const SlVec3,
    satellites: *const SlVec3,
    n: usize,
    out: *mut SlDop,
) -> i32 {
    guard(|| {
        if receiver.is_null() || satellites.is_null() || out.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_dilution_of_precision: null pointer",
            );
        }
        // SAFETY: checked non-null; the caller states `satellites` is readable for `n`.
        let (rx, sats) = unsafe {
            (
                *receiver,
                core::slice::from_raw_parts(satellites, n).to_vec(),
            )
        };
        let rx = [rx.x, rx.y, rx.z];
        let sats: Vec<[f64; 3]> = sats.iter().map(|s| [s.x, s.y, s.z]).collect();
        let Some(d) = gnss_dop(rx, &sats) else {
            return fail(
                SL_ERR_DOMAIN,
                format!(
                    "sl_dilution_of_precision: {n} satellites give no solution -- four is the \
                     minimum, and they must not be coplanar with the receiver"
                ),
            );
        };
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe {
            *out = SlDop {
                gdop: d.gdop,
                pdop: d.pdop,
                hdop: d.hdop,
                vdop: d.vdop,
                tdop: d.tdop,
            }
        };
        SL_OK
    })
}

/// Earth's magnetic field at an ECEF position, tesla, from IGRF-14 degree and order 13.
///
/// # Safety
/// `position_m` points at one `sl_vec3`; `out_tesla` is writable for one `sl_vec3`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_igrf_field_ecef(
    position_m: *const SlVec3,
    year: f64,
    out_tesla: *mut SlVec3,
) -> i32 {
    guard(|| {
        if position_m.is_null() || out_tesla.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, "sl_igrf_field_ecef: null pointer");
        }
        // SAFETY: checked non-null just above.
        let r = unsafe { *position_m };
        const WHO: &str = "sl_igrf_field_ecef";
        if let Err(status) = check_all_finite(WHO, "position_m", &[r.x, r.y, r.z])
            .and_then(|()| check_finite(WHO, "year", year))
        {
            return status;
        }
        let b = igrf_field_ecef([r.x, r.y, r.z], year);
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe {
            *out_tesla = SlVec3 {
                x: b[0],
                y: b[1],
                z: b[2],
            }
        };
        SL_OK
    })
}

// ── Ionosphere ────────────────────────────────────────────────────────────────────────────────

/// The broadcast (Klobuchar) L1 ionospheric slant delay, metres.
///
/// # Safety
/// `alpha` and `beta` are each readable for four `double`; `out_metres` is writable for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_klobuchar_iono_delay(
    alpha: *const f64,
    beta: *const f64,
    latitude_rad: f64,
    longitude_rad: f64,
    elevation_rad: f64,
    azimuth_rad: f64,
    gps_seconds_of_day: f64,
    out_metres: *mut f64,
) -> i32 {
    guard(|| {
        if alpha.is_null() || beta.is_null() || out_metres.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_klobuchar_iono_delay: null pointer",
            );
        }
        // SAFETY: the caller states each coefficient array is readable for four doubles.
        let (a, b) = unsafe { (*alpha.cast::<[f64; 4]>(), *beta.cast::<[f64; 4]>()) };
        let delay = gnss_klobuchar(
            a,
            b,
            latitude_rad,
            longitude_rad,
            elevation_rad,
            azimuth_rad,
            gps_seconds_of_day,
        );
        // SAFETY: checked non-null just above.
        unsafe { *out_metres = delay };
        SL_OK
    })
}

/// The first-order ionospheric range delay, `40.3 * TEC / f^2`, metres. Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_ionospheric_path_delay(stec_tecu: f64, frequency_hz: f64) -> f64 {
    iono_path_delay(stec_tecu, frequency_hz)
}

/// The second-order ionospheric group delay, metres -- the geomagnetic `1/f^3` term that survives
/// an ionosphere-free combination. Signed. Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_second_order_ionospheric_delay(
    latitude_rad: f64,
    longitude_rad: f64,
    elevation_rad: f64,
    azimuth_rad: f64,
    stec_tecu: f64,
    frequency_hz: f64,
    pierce_height_m: f64,
) -> f64 {
    gnss_hoi2(
        latitude_rad,
        longitude_rad,
        elevation_rad,
        azimuth_rad,
        pierce_height_m,
        stec_tecu * 1e16,
        frequency_hz,
    )
}

/// The thin-shell obliquity factor that maps a vertical TEC to a slant one. Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_ionospheric_mapping_factor(
    elevation_rad: f64,
    ionosphere_height_m: f64,
    earth_radius_m: f64,
) -> f64 {
    gnss_iono_mapping(elevation_rad, ionosphere_height_m, earth_radius_m)
}

/// The NeQuick-G (Galileo) slant total electron content, TECU.
///
/// # Safety
/// `alpha` is readable for three `double`; `receiver` and `satellite` each point at one
/// `sl_geodetic`; `out_tecu` is writable for one `double`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_nequick_galileo_stec(
    alpha: *const f64,
    month: u32,
    seconds_in_utc_day: f64,
    receiver: *const SlGeodetic,
    satellite: *const SlGeodetic,
    out_tecu: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_nequick_galileo_stec";
        if alpha.is_null() || receiver.is_null() || satellite.is_null() || out_tecu.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: the caller states `alpha` is readable for three doubles.
        let a = unsafe { *alpha.cast::<[f64; 3]>() };
        // SAFETY: checked non-null just above.
        let (rx, sat) = unsafe { (*receiver, *satellite) };
        if let Err(status) = check_stec(WHO, month, seconds_in_utc_day, rx, sat) {
            return status;
        }
        let stec = NeQuickGalileo::new(a).stec(
            DateTimeComponents::new(month, seconds_in_utc_day),
            geodetic_point(rx),
            geodetic_point(sat),
        );
        // SAFETY: checked non-null just above.
        unsafe { *out_tecu = stec };
        SL_OK
    })
}

fn geodetic_point(g: SlGeodetic) -> GeodeticPoint {
    GeodeticPoint::new(g.latitude_rad, g.longitude_rad, g.height_m)
}

/// The epoch and the two endpoints both STEC entry points share, checked once.
fn check_stec(
    who: &str,
    month: u32,
    seconds_in_utc_day: f64,
    rx: SlGeodetic,
    sat: SlGeodetic,
) -> Result<(), i32> {
    if !(1..=12).contains(&month) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: month {month} is not in 1..=12"),
        ));
    }
    if !(seconds_in_utc_day.is_finite() && (0.0..=86_400.0).contains(&seconds_in_utc_day)) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: seconds_in_utc_day must be finite and within one day"),
        ));
    }
    for (name, g) in [("receiver", rx), ("satellite", sat)] {
        if !(g.latitude_rad.is_finite() && g.longitude_rad.is_finite() && g.height_m.is_finite()) {
            return Err(fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{who}: the {name} point must be finite"),
            ));
        }
    }
    Ok(())
}

// ── Troposphere ───────────────────────────────────────────────────────────────────────────────

/// The Saastamoinen zenith hydrostatic delay, metres, from surface pressure (hPa), geodetic
/// latitude (radians) and ellipsoidal height (metres). Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_saastamoinen_zenith_hydrostatic(
    pressure_hpa: f64,
    latitude_rad: f64,
    height_m: f64,
) -> f64 {
    gnss_saastamoinen_hydrostatic(pressure_hpa, latitude_rad, height_m)
}

/// The Saastamoinen zenith wet delay, metres, from surface temperature (K) and water-vapour
/// partial pressure (hPa). Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_saastamoinen_zenith_wet(temperature_k: f64, vapour_pressure_hpa: f64) -> f64 {
    gnss_saastamoinen_wet(temperature_k, vapour_pressure_hpa)
}

/// Write a mapping pair, or say what is wrong with the request. The four mapping functions differ
/// only in what they read, so they share their checking and their delivery.
/// A mapping function scales a zenith delay along a line of sight, so it needs a line of sight that
/// leaves the antenna: at or below the horizon there is no atmosphere path to scale, and the
/// seasonal term needs a day of the year to evaluate at.
fn check_mapping_geometry(who: &str, elevation_rad: f64, day_of_year: f64) -> Result<(), i32> {
    if !(elevation_rad.is_finite() && elevation_rad > 0.0) {
        return Err(fail(
            SL_ERR_DOMAIN,
            format!("{who}: a mapping function is defined above the horizon, not at or below it"),
        ));
    }
    check_finite(who, "day_of_year", day_of_year)
}

unsafe fn deliver_mapping(
    who: &str,
    elevation_rad: f64,
    day_of_year: f64,
    factors: impl FnOnce() -> (f64, f64),
    out: *mut SlMapping,
) -> i32 {
    if out.is_null() {
        return fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: null out"));
    }
    if let Err(status) = check_mapping_geometry(who, elevation_rad, day_of_year) {
        return status;
    }
    let (hydrostatic, wet) = factors();
    // SAFETY: checked non-null; written only now that the answer exists.
    unsafe { *out = SlMapping { hydrostatic, wet } };
    SL_OK
}

/// The Niell mapping functions.
///
/// # Safety
/// `out` is writable for one `sl_mapping`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_niell_mapping(
    latitude_rad: f64,
    height_m: f64,
    day_of_year: f64,
    elevation_rad: f64,
    out: *mut SlMapping,
) -> i32 {
    guard(|| {
        // SAFETY: forwarding the caller's own contract on `out`.
        unsafe {
            deliver_mapping(
                "sl_niell_mapping",
                elevation_rad,
                day_of_year,
                || gnss_niell(latitude_rad, height_m, day_of_year, elevation_rad),
                out,
            )
        }
    })
}

/// The Global Mapping Function.
///
/// # Safety
/// `out` is writable for one `sl_mapping`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_gmf_mapping(
    latitude_rad: f64,
    longitude_rad: f64,
    height_m: f64,
    day_of_year: f64,
    elevation_rad: f64,
    out: *mut SlMapping,
) -> i32 {
    guard(|| {
        // SAFETY: forwarding the caller's own contract on `out`.
        unsafe {
            deliver_mapping(
                "sl_gmf_mapping",
                elevation_rad,
                day_of_year,
                || {
                    gnss_gmf(
                        latitude_rad,
                        longitude_rad,
                        height_m,
                        day_of_year,
                        elevation_rad,
                    )
                },
                out,
            )
        }
    })
}

/// The Vienna Mapping Function 1, from its two `a` coefficients.
///
/// # Safety
/// `out` is writable for one `sl_mapping`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_vmf1_mapping(
    ah: f64,
    aw: f64,
    latitude_rad: f64,
    height_m: f64,
    day_of_year: f64,
    elevation_rad: f64,
    out: *mut SlMapping,
) -> i32 {
    guard(|| {
        // SAFETY: forwarding the caller's own contract on `out`.
        unsafe {
            deliver_mapping(
                "sl_vmf1_mapping",
                elevation_rad,
                day_of_year,
                || gnss_vmf1(ah, aw, latitude_rad, height_m, day_of_year, elevation_rad),
                out,
            )
        }
    })
}

/// The Vienna Mapping Function 3.
///
/// # Safety
/// `out` is writable for one `sl_mapping`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_vmf3_mapping(
    ah: f64,
    aw: f64,
    latitude_rad: f64,
    longitude_rad: f64,
    height_m: f64,
    day_of_year: f64,
    elevation_rad: f64,
    out: *mut SlMapping,
) -> i32 {
    guard(|| {
        // SAFETY: forwarding the caller's own contract on `out`.
        unsafe {
            deliver_mapping(
                "sl_vmf3_mapping",
                elevation_rad,
                day_of_year,
                || {
                    gnss_vmf3(
                        ah,
                        aw,
                        latitude_rad,
                        longitude_rad,
                        height_m,
                        day_of_year,
                        elevation_rad,
                    )
                },
                out,
            )
        }
    })
}

// ── Atmospheres, tides, density calibration and space weather ─────────────────────────────────

/// `sl_mean_pole`: IERS 2010 Chapter 7 as amended 2018/02/01 — the secular pole.
const MEAN_POLE_SECULAR2018: i32 = 0;
/// `sl_mean_pole`: the retired Table 7.7 mean pole, which is what Orekit implements.
const MEAN_POLE_IERS2010: i32 = 1;

/// How many coefficients a degree-4 density correction carries.
const DENSITY_CORRECTION_COEFFICIENTS: usize = 25;

/// How many drivers the density regression is built over.
const DENSITY_DRIVERS: usize = 5;

/// How many entries the NRLMSISE-00 storm-time `ap_array` holds.
const AP_ARRAY_LEN: usize = 7;

/// `sl_zonal`: the three low-degree normalized zonal geopotential coefficients.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlZonal {
    /// Degree 2.
    pub c20: f64,
    /// Degree 3.
    pub c30: f64,
    /// Degree 4.
    pub c40: f64,
}

/// `sl_drag_panel`: Sentman's four coefficients for one flat panel.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlDragPanel {
    /// Normal pressure coefficient.
    pub pressure: f64,
    /// Tangential shear coefficient.
    pub shear: f64,
    /// Coefficient along the incident flow.
    pub drag: f64,
    /// Coefficient across it.
    pub lift: f64,
}

/// `sl_swath`: the geometry one look angle cuts out of the ground.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlSwath {
    /// Earth-central angle from the sub-satellite point to the target, radians.
    pub earth_central_angle_rad: f64,
    /// Elevation of the spacecraft seen from the target, radians.
    pub elevation_rad: f64,
    /// Along-ground distance to the target, metres.
    pub ground_range_m: f64,
    /// Line-of-sight distance to it, metres.
    pub slant_range_m: f64,
}

/// `sl_density_sample`: one density-ratio observation the calibration is fitted to.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlDensitySample {
    /// Geocentric latitude, radians.
    pub latitude_rad: f64,
    /// Local solar time, hours in `[0, 24)`.
    pub local_solar_time_h: f64,
    /// `ln(ρ_measured / ρ_model)`.
    pub log_ratio: f64,
    /// The weight this sample carries in the normal equations.
    pub weight: f64,
}

/// `sl_density_observation`: one height-normalized absolute log-density from a density file.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlDensityObservation {
    /// Calendar year of the observation.
    pub year: i32,
    /// Calendar month, 1-12.
    pub month: u32,
    /// Calendar day, 1-31.
    pub day: u32,
    /// Geocentric latitude, radians.
    pub latitude_rad: f64,
    /// Local solar time, hours in `[0, 24)`.
    pub local_solar_time_h: f64,
    /// `ln ρ` lifted to the reference altitude.
    pub ln_density_ref: f64,
}

/// `sl_space_weather_day`: one day of a CelesTrak `SW-All.csv`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlSpaceWeatherDay {
    /// Calendar year.
    pub year: i32,
    /// Calendar month, 1-12.
    pub month: u32,
    /// Calendar day, 1-31.
    pub day: u32,
    /// The daily planetary amplitude.
    pub ap: f64,
    /// Observed 10.7 cm solar flux, sfu.
    pub f107_observed: f64,
    /// Its 81-day centred mean, sfu.
    pub f107_center81: f64,
}

/// `sl_brouwer_lyddane_params`: the field the mean-element theory is built over.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlBrouwerLyddaneParams {
    /// The osculating state at the epoch, metres and metres per second.
    pub initial: SlState,
    /// Gravitational parameter, m³/s².
    pub mu: f64,
    /// Equatorial radius the coefficients are normalized to, metres.
    pub earth_radius_m: f64,
    /// Normalized zonal coefficient of degree 2.
    pub c20: f64,
    /// Degree 3.
    pub c30: f64,
    /// Degree 4.
    pub c40: f64,
    /// Degree 5.
    pub c50: f64,
    /// The drag-like secular mean-motion rate, rad/s².
    pub m2: f64,
}

/// The mean pole an `sl_mean_pole` tag names.
fn mean_pole_of(who: &str, tag: i32) -> Result<MeanPole, i32> {
    match tag {
        MEAN_POLE_SECULAR2018 => Ok(MeanPole::Secular2018),
        MEAN_POLE_IERS2010 => Ok(MeanPole::Iers2010),
        _ => Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: mean pole tag {tag} is not one of sl_mean_pole"),
        )),
    }
}

/// The solid and ocean pole tides differ only in which correction they evaluate, and §6.4 and §6.5
/// share the definition of the wobble variables, so they share their checking too.
///
/// # Safety
/// `out` is writable for two doubles.
unsafe fn deliver_pole_tide(
    who: &str,
    wobble: [f64; 3],
    mean_pole: i32,
    corrections: impl FnOnce(f64, f64, f64, MeanPole) -> [f64; 2],
    out: *mut f64,
) -> i32 {
    if out.is_null() {
        return fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: null out pointer"));
    }
    if let Err(status) = check_all_finite(who, "the wobble (xp, yp, t)", &wobble) {
        return status;
    }
    let pole = match mean_pole_of(who, mean_pole) {
        Ok(pole) => pole,
        Err(status) => return status,
    };
    let c = corrections(wobble[0], wobble[1], wobble[2], pole);
    // SAFETY: checked non-null, and the caller states it is writable for two doubles.
    unsafe { core::ptr::copy_nonoverlapping(c.as_ptr(), out, 2) };
    SL_OK
}

/// The three zonal answers share their shape, so they share their delivery.
///
/// # Safety
/// `out` is writable for one `sl_zonal`.
unsafe fn deliver_zonal(who: &str, years_since_j2000: f64, c: [f64; 3], out: *mut SlZonal) -> i32 {
    if out.is_null() {
        return fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: null out pointer"));
    }
    if let Err(status) = check_finite(who, "years_since_j2000", years_since_j2000) {
        return status;
    }
    // SAFETY: checked non-null.
    unsafe {
        *out = SlZonal {
            c20: c[0],
            c30: c[1],
            c40: c[2],
        }
    };
    SL_OK
}

/// The samples a caller's array becomes.
///
/// # Safety
/// `samples` is readable for `n` `sl_density_sample`.
unsafe fn borrow_density_samples(samples: *const SlDensitySample, n: usize) -> Vec<DensitySample> {
    // SAFETY: the caller states `samples` is readable for `n` elements.
    let rows = unsafe { core::slice::from_raw_parts(samples, n) };
    rows.iter()
        .map(|s| DensitySample {
            latitude: s.latitude_rad,
            local_solar_time: s.local_solar_time_h,
            log_ratio: s.log_ratio,
            weight: s.weight,
        })
        .collect()
}

/// NRLMSISE-00 total mass density, kg/m³, at one place and time. Total.
///
/// `sec` is UT seconds into the day; the local solar time the model needs is derived from it and
/// the longitude, exactly as the Python bridge does, so the two answer the same number.
///
/// A non-finite argument answers NaN. The published subroutine does not propagate one — measured
/// here, a NaN altitude comes back as 2.9e-3 kg/m³, which is a sea-level density and a plausible
/// number rather than an obvious failure. Returning NaN is the honest answer a total function can
/// give, and the check belongs at this boundary because it is the one place a caller's arithmetic
/// stops being Rust's problem.
#[allow(clippy::too_many_arguments)] // a day, a time, three coordinates and three solar-activity indices
#[unsafe(no_mangle)]
pub extern "C" fn sl_nrlmsise00_density(
    doy: i32,
    sec: f64,
    altitude_km: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    f107: f64,
    f107a: f64,
    ap: f64,
) -> f64 {
    guard_f64(|| {
        let arguments = [
            sec,
            altitude_km,
            latitude_deg,
            longitude_deg,
            f107,
            f107a,
            ap,
        ];
        if arguments.iter().any(|v| !v.is_finite()) {
            return f64::NAN;
        }
        let input = Nrlmsise00Input {
            year: 0,
            doy,
            sec,
            alt: altitude_km,
            g_lat: latitude_deg,
            g_long: longitude_deg,
            lst: sec / 3600.0 + longitude_deg / 15.0,
            f107a,
            f107,
            ap,
            ap_array: None,
        };
        gtd7d(&input, &Nrlmsise00Flags::default()).d[5] * 1000.0 // g/cm³ → kg/m³
    })
}

/// How many doubles `sl_sdtm1_density_interval` writes.
pub const SL_SDTM1_INTERVAL: usize = 2;

/// The evaluation point the `sl_sdtm1_*` point functions share, assembled the way the fit's own data
/// was: a fractional day of year, and local solar time from UT and longitude wrapped into a day.
fn sdtm1_point(
    doy: f64,
    sec: f64,
    altitude_km: f64,
    latitude_deg: f64,
    longitude_deg: f64,
) -> SdtmPoint {
    let mut lst = sec / 3600.0 + longitude_deg / 15.0;
    lst -= 24.0 * (lst / 24.0).floor();
    SdtmPoint {
        altitude: altitude_km * 1000.0,
        latitude: latitude_deg.to_radians(),
        local_solar_time: lst,
        day_of_year: doy + sec / 86_400.0,
    }
}

/// SDTM-1 total mass density, kg/m³, at one place and time — this library's own thermosphere model.
///
/// `sec` is UT seconds into the day; the local solar time the model needs is derived from it and the
/// longitude, exactly as the Python bridge does, so the two answer the same number. `f107` is the
/// **previous day's** observed flux and `f107a` its 81-day **trailing** mean, which is the fit's
/// convention and not NRLMSISE-00's centred one. `storm` is `sl_sdtm1_storm_response`'s output over
/// the `ap` history — not an `ap` value.
///
/// A non-finite argument answers NaN, at this boundary rather than inside the model: the Bates profile
/// would carry a NaN altitude through to a NaN density on its own, but a NaN day of year reaches only
/// the seasonal harmonics and would come back as a plausible mid-season number.
#[allow(clippy::too_many_arguments)] // a day, a time, three coordinates and three drivers
#[unsafe(no_mangle)]
pub extern "C" fn sl_sdtm1_density(
    doy: f64,
    sec: f64,
    altitude_km: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    f107: f64,
    f107a: f64,
    storm: f64,
) -> f64 {
    guard_f64(|| {
        let arguments = [
            doy,
            sec,
            altitude_km,
            latitude_deg,
            longitude_deg,
            f107,
            f107a,
            storm,
        ];
        if arguments.iter().any(|v| !v.is_finite()) {
            return f64::NAN;
        }
        SDTM1_FIELD.density(
            &sdtm1_point(doy, sec, altitude_km, latitude_deg, longitude_deg),
            &SdtmDrivers { f107, f107a, storm },
        )
    })
}

/// SDTM-1 exospheric temperature `T∞`, K, at the same point and drivers as `sl_sdtm1_density`.
///
/// The quantity the fit parameterises — the density is a Bates profile above it — so this is the
/// handle for comparing against a published exospheric temperature rather than against a density. A
/// non-finite argument answers NaN.
#[allow(clippy::too_many_arguments)] // a day, a time, three coordinates and three drivers
#[unsafe(no_mangle)]
pub extern "C" fn sl_sdtm1_exospheric_temperature(
    doy: f64,
    sec: f64,
    altitude_km: f64,
    latitude_deg: f64,
    longitude_deg: f64,
    f107: f64,
    f107a: f64,
    storm: f64,
) -> f64 {
    guard_f64(|| {
        let arguments = [
            doy,
            sec,
            altitude_km,
            latitude_deg,
            longitude_deg,
            f107,
            f107a,
            storm,
        ];
        if arguments.iter().any(|v| !v.is_finite()) {
            return f64::NAN;
        }
        SDTM1_FIELD.exospheric_temperature(
            &sdtm1_point(doy, sec, altitude_km, latitude_deg, longitude_deg),
            &SdtmDrivers { f107, f107a, storm },
        )
    })
}

/// The storm response over a borrowed history — the predicate in a safe function, where
/// `cargo mutants` can reach it.
fn sdtm1_storm_response(history: &[f64], cadence_s: f64) -> f64 {
    if !cadence_s.is_finite() || cadence_s <= 0.0 {
        return f64::NAN;
    }
    SDTM1_FIELD.storm().response(history, cadence_s)
}

/// SDTM-1's geomagnetic storm driver: the shipped heating and nitric-oxide cooling state run over an
/// index history.
///
/// `history` is the `ap` series oldest first, ending at the interval **before** the sample, and
/// `cadence_s` is that series' step — 10 800 s for CelesTrak's 3-hourly `ap`, which is what the fit
/// used. The three constants belong to the fitted field, so a caller cannot build this driver
/// itself. Nitric oxide remembers a storm for days, and a truncated history reads as a different
/// storm: pass fifteen days, 120 intervals (`sdtm::coefficients::HISTORY` in Rust).
///
/// `n = 0` answers `0`, the response of a state with nothing to respond to, which is what the Python
/// bridge answers for an empty list. A null pointer answers NaN: in C that is a different thing from
/// an empty array, and the only honest answer a function returning a double can give.
///
/// # Safety
/// `history` is readable for `n` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sdtm1_storm_response(
    history: *const f64,
    n: usize,
    cadence_s: f64,
) -> f64 {
    guard_f64(|| {
        if history.is_null() {
            return f64::NAN;
        }
        // SAFETY: checked non-null; the caller states it is readable for `n` doubles.
        let series = unsafe { core::slice::from_raw_parts(history, n) };
        sdtm1_storm_response(series, cadence_s)
    })
}

/// SDTM-1's log-normal density interval about `density`, `[lower, upper]` in kg/m³.
///
/// The spread of `ln(ρ_model/ρ_measured)` was measured by flux × storm regime on the residual of a fit
/// with the calendar month held out, so this is the model's error against measurement rather than a
/// propagation of parameter covariance. `z` is the standard-normal multiplier: 1.645 for 90 %, 1.960
/// for 95 %, 2.576 for 99 %. Which regime applies is decided by `f107a` and `storm`, so they must be
/// the values `density` was evaluated with.
///
/// # Safety
/// `out` is writable for `SL_SDTM1_INTERVAL` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sdtm1_density_interval(
    density: f64,
    f107: f64,
    f107a: f64,
    storm: f64,
    z: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_sdtm1_density_interval";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out pointer"));
        }
        let (lower, upper) =
            SDTM1_UNCERTAINTY.interval(density, &SdtmDrivers { f107, f107a, storm }, z);
        // SAFETY: checked non-null; the caller states it is writable for two doubles.
        unsafe {
            core::ptr::copy_nonoverlapping([lower, upper].as_ptr(), out, SL_SDTM1_INTERVAL);
        }
        SL_OK
    })
}

/// Brouwer–Lyddane analytical propagation with the Lyddane drag term: the osculating state at each
/// requested offset from the epoch.
///
/// # Safety
/// `p` points at one initialised `sl_brouwer_lyddane_params`; `times_seconds` is readable for `n`
/// doubles and `out` writable for `n` `sl_state`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_brouwer_lyddane_propagate(
    p: *const SlBrouwerLyddaneParams,
    times_seconds: *const f64,
    n: usize,
    out: *mut SlState,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_brouwer_lyddane_propagate";
        if p.is_null() || times_seconds.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let p = unsafe { *p };
        if let Err(status) = check_positive(WHO, "mu", p.mu, SL_ERR_DOMAIN) {
            return status;
        }
        if let Err(status) = check_positive(WHO, "earth_radius_m", p.earth_radius_m, SL_ERR_DOMAIN)
        {
            return status;
        }
        // SAFETY: the caller states `times_seconds` is readable for `n` doubles.
        let times = unsafe { core::slice::from_raw_parts(times_seconds, n) };
        if let Err(status) = check_all_finite(WHO, "times_seconds", times) {
            return status;
        }
        let osculating = KeplerianElements::from_state(
            to_vec3(p.initial.position),
            to_vec3(p.initial.velocity),
            p.mu,
        );
        let model = match BrouwerLyddane::from_osculating_with_drag(
            &osculating,
            p.earth_radius_m,
            p.c20,
            p.c30,
            p.c40,
            p.c50,
            p.m2,
        ) {
            Ok(model) => model,
            Err(e) => return fail(SL_ERR_DOMAIN, format!("{WHO}: {e}")),
        };
        // Every epoch is checked before any of them is written, for the reason `dsst_sample` states.
        for (i, &t) in times.iter().enumerate() {
            let (r, v) = model.propagate(t);
            // SAFETY: the caller states `out` is writable for `n` states, and `i < n`.
            unsafe { *out.add(i) = from_state(r, v) };
        }
        SL_OK
    })
}

/// IERS 2010 solid-tide Step-2 frequency corrections: the five `[ΔC20, ΔC21, ΔS21, ΔC22, ΔS22]`
/// from the Greenwich mean sidereal time and the five Delaunay arguments, all radians.
///
/// # Safety
/// `delaunay` is readable for five doubles and `out` writable for five.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_solid_tide_step2(
    gmst_rad: f64,
    delaunay: *const f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_solid_tide_step2";
        if delaunay.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_finite(WHO, "gmst_rad", gmst_rad) {
            return status;
        }
        let mut arguments = [0.0_f64; 5];
        // SAFETY: checked non-null; the caller states it is readable for five doubles.
        arguments.copy_from_slice(unsafe { core::slice::from_raw_parts(delaunay, 5) });
        if let Err(status) = check_all_finite(WHO, "delaunay", &arguments) {
            return status;
        }
        let c = step2_corrections(gmst_rad, arguments);
        // SAFETY: checked non-null; the caller states it is writable for five doubles.
        unsafe { core::ptr::copy_nonoverlapping(c.as_ptr(), out, 5) };
        SL_OK
    })
}

/// IERS 2010 §6.4 solid pole tide: the `[ΔC21, ΔS21]` from the polar motion, radians, at `t`
/// seconds since J2000 (TT).
///
/// # Safety
/// `out` is writable for two doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_solid_pole_tide(
    xp_rad: f64,
    yp_rad: f64,
    t_seconds: f64,
    mean_pole: i32,
    out: *mut f64,
) -> i32 {
    guard(|| {
        // SAFETY: forwarding the caller's own contract on `out`.
        unsafe {
            deliver_pole_tide(
                "sl_solid_pole_tide",
                [xp_rad, yp_rad, t_seconds],
                mean_pole,
                pole_tide_corrections,
                out,
            )
        }
    })
}

/// IERS 2010 §6.5 ocean pole tide: the `[ΔC21, ΔS21]` that superposes with the solid pole tide on
/// the same band, and so must be given the same `mean_pole`.
///
/// # Safety
/// `out` is writable for two doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ocean_pole_tide(
    xp_rad: f64,
    yp_rad: f64,
    t_seconds: f64,
    mean_pole: i32,
    out: *mut f64,
) -> i32 {
    guard(|| {
        // SAFETY: forwarding the caller's own contract on `out`.
        unsafe {
            deliver_pole_tide(
                "sl_ocean_pole_tide",
                [xp_rad, yp_rad, t_seconds],
                mean_pole,
                ocean_pole_tide_corrections,
                out,
            )
        }
    })
}

/// The IERS 2010 conventional zonal coefficients at `years_since_j2000`: mean plus secular drift,
/// C20 zero-tide.
///
/// # Safety
/// `out` is writable for one `sl_zonal`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_conventional_zonal_coefficients(
    years_since_j2000: f64,
    out: *mut SlZonal,
) -> i32 {
    guard(|| {
        // SAFETY: forwarding the caller's own contract on `out`.
        unsafe {
            deliver_zonal(
                "sl_conventional_zonal_coefficients",
                years_since_j2000,
                tvg_conventional(years_since_j2000),
                out,
            )
        }
    })
}

/// The secular corrections alone — add them to a base field's C20/C30/C40 for time-variable gravity.
///
/// # Safety
/// `out` is writable for one `sl_zonal`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_time_variable_zonal_correction(
    years_since_j2000: f64,
    out: *mut SlZonal,
) -> i32 {
    guard(|| {
        // SAFETY: forwarding the caller's own contract on `out`.
        unsafe {
            deliver_zonal(
                "sl_time_variable_zonal_correction",
                years_since_j2000,
                tvg_secular(years_since_j2000),
                out,
            )
        }
    })
}

/// Sentman's free-molecular coefficients for one flat panel.
///
/// # Safety
/// `out` is writable for one `sl_drag_panel`.
#[allow(clippy::too_many_arguments)] // the flow, the geometry, the accommodation and two temperatures
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sentman_drag_coefficients(
    speed_ratio: f64,
    cos_incidence: f64,
    accommodation: f64,
    wall_temperature_k: f64,
    atmosphere_temperature_k: f64,
    out: *mut SlDragPanel,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_sentman_drag_coefficients";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out pointer"));
        }
        if let Err(status) =
            check_positive(WHO, "wall_temperature_k", wall_temperature_k, SL_ERR_DOMAIN).and_then(
                |()| {
                    check_positive(
                        WHO,
                        "atmosphere_temperature_k",
                        atmosphere_temperature_k,
                        SL_ERR_DOMAIN,
                    )
                },
            )
        {
            return status;
        }
        let c = sentman_panel(
            speed_ratio,
            cos_incidence,
            accommodation,
            wall_temperature_k,
            atmosphere_temperature_k,
        );
        // SAFETY: checked non-null.
        unsafe {
            *out = SlDragPanel {
                pressure: c.pressure,
                shear: c.shear,
                drag: c.drag,
                lift: c.lift,
            }
        };
        SL_OK
    })
}

/// The molecular speed ratio `s` the free-molecular coefficients are parameterised by. Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_molecular_speed_ratio(
    relative_speed_ms: f64,
    temperature_k: f64,
    mean_molar_mass: f64,
) -> f64 {
    guard_f64(|| sentman_speed_ratio(relative_speed_ms, temperature_k, mean_molar_mass))
}

/// The nonlinear density drivers a regression is built over: `[flux, ap, flux², √ap, 1]`, five out.
///
/// # Safety
/// `out` is writable for five doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_space_weather_drivers(
    f107: f64,
    f107_81: f64,
    ap: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_space_weather_drivers: null out pointer",
            );
        }
        let d = density_drivers(f107, f107_81, ap);
        // SAFETY: checked non-null; the caller states it is writable for five doubles.
        unsafe { core::ptr::copy_nonoverlapping(d.as_ptr(), out, DENSITY_DRIVERS) };
        SL_OK
    })
}

/// The seven-element NRLMSISE-00 storm-time `ap_array` for one day and time, built from the
/// 3-hourly history in a CelesTrak `SW-All.csv`.
///
/// The day and its ~2.5-day look-back must both be in the file; `SL_ERR_DOMAIN` says they are not.
///
/// # Safety
/// `csv_text` is readable for `text_len` bytes and `out` writable for seven doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_space_weather_ap_array(
    csv_text: *const c_char,
    text_len: usize,
    year: i32,
    month: u32,
    day: u32,
    seconds_of_day: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_space_weather_ap_array";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out pointer"));
        }
        // SAFETY: the caller states `csv_text` is readable for `text_len`.
        let Some(text) = (unsafe { borrow_sized_str(csv_text, text_len) }) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: csv_text is null or not UTF-8"),
            );
        };
        let days = parse_space_weather(text);
        let Some(ap) = nrlmsise_ap_array(&days, year, month, day, seconds_of_day) else {
            return fail(
                SL_ERR_DOMAIN,
                format!(
                    "{WHO}: {year}-{month:02}-{day:02} or its 2.5-day look-back is not in the \
                     {} days the file holds",
                    days.len()
                ),
            );
        };
        // SAFETY: checked non-null; the caller states it is writable for seven doubles.
        unsafe { core::ptr::copy_nonoverlapping(ap.as_ptr(), out, AP_ARRAY_LEN) };
        SL_OK
    })
}

/// Every day a CelesTrak `SW-All.csv` holds. Two-call protocol; see the header.
///
/// # Safety
/// `csv_text` is readable for `text_len` bytes; `out` is null or writable for `cap`
/// `sl_space_weather_day`; `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_space_weather_days(
    csv_text: *const c_char,
    text_len: usize,
    out: *mut SlSpaceWeatherDay,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: the caller states `csv_text` is readable for `text_len`.
        let Some(text) = (unsafe { borrow_sized_str(csv_text, text_len) }) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_space_weather_days: csv_text is null or not UTF-8",
            );
        };
        let days: Vec<SlSpaceWeatherDay> = parse_space_weather(text)
            .into_iter()
            .map(|d| SlSpaceWeatherDay {
                year: d.year,
                month: d.month,
                day: d.day,
                ap: d.ap,
                f107_observed: d.f107_observed,
                f107_center81: d.f107_center81,
            })
            .collect();
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&days, out, cap, out_len) }
    })
}

/// The orbit-mean density-ratio samples a TU Delft density file reduces to. Two-call protocol.
///
/// # Safety
/// `text` is readable for `text_len` bytes; `out` is null or writable for `cap`
/// `sl_density_sample`; `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_density_data_samples(
    text: *const c_char,
    text_len: usize,
    out: *mut SlDensitySample,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: the caller states `text` is readable for `text_len`.
        let Some(text) = (unsafe { borrow_sized_str(text, text_len) }) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_density_data_samples: text is null or not UTF-8",
            );
        };
        let samples: Vec<SlDensitySample> = orbit_mean_samples(&parse_density(text))
            .into_iter()
            .map(|s| SlDensitySample {
                latitude_rad: s.latitude,
                local_solar_time_h: s.local_solar_time,
                log_ratio: s.log_ratio,
                weight: s.weight,
            })
            .collect();
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&samples, out, cap, out_len) }
    })
}

/// Height-normalized absolute log-densities from the same file: `ln ρ` lifted to
/// `reference_altitude_m` through an exponential `scale_height_m`, for every nominal observation.
/// Unlike `sl_density_data_samples` this keeps the absolute amplitude — the signal that tracks
/// solar flux. Two-call protocol.
///
/// # Safety
/// `text` is readable for `text_len` bytes; `out` is null or writable for `cap`
/// `sl_density_observation`; `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_density_altitude_normalized(
    text: *const c_char,
    text_len: usize,
    reference_altitude_m: f64,
    scale_height_m: f64,
    out: *mut SlDensityObservation,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_density_altitude_normalized";
        // SAFETY: the caller states `text` is readable for `text_len`.
        let Some(text) = (unsafe { borrow_sized_str(text, text_len) }) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: text is null or not UTF-8"),
            );
        };
        if let Err(status) = check_finite(WHO, "reference_altitude_m", reference_altitude_m) {
            return status;
        }
        if let Err(status) = check_positive(WHO, "scale_height_m", scale_height_m, SL_ERR_DOMAIN) {
            return status;
        }
        let rows: Vec<SlDensityObservation> = parse_density(text)
            .into_iter()
            .filter(|o| o.nominal && o.density > 0.0)
            .map(|o| SlDensityObservation {
                year: o.year,
                month: o.month,
                day: o.day,
                latitude_rad: o.latitude,
                local_solar_time_h: o.local_solar_time,
                ln_density_ref: o.normalized_log_density(reference_altitude_m, scale_height_m),
            })
            .collect();
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&rows, out, cap, out_len) }
    })
}

/// The multiplicative density correction a fitted coefficient vector gives at one place and time.
///
/// # Safety
/// `coefficients` is readable for `n` doubles and `out` writable for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_density_correction_factor(
    coefficients: *const f64,
    n: usize,
    latitude_rad: f64,
    local_solar_time_h: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_density_correction_factor";
        if coefficients.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_count(WHO, "n", n, DENSITY_CORRECTION_COEFFICIENTS) {
            return status;
        }
        let mut coeffs = [0.0_f64; DENSITY_CORRECTION_COEFFICIENTS];
        // SAFETY: checked non-null and the length checked equal; the caller states it is readable.
        coeffs.copy_from_slice(unsafe { core::slice::from_raw_parts(coefficients, n) });
        let factor =
            DensityCorrection::<DENSITY_CORRECTION_COEFFICIENTS>::from_coefficients(coeffs)
                .factor(latitude_rad, local_solar_time_h);
        // SAFETY: checked non-null.
        unsafe { *out = factor };
        SL_OK
    })
}

/// Fit that coefficient vector by ridge regression over the samples. Two-call protocol; the result
/// is always `SL_DENSITY_CORRECTION_COEFFICIENTS` long.
///
/// # Safety
/// `samples` is readable for `n` `sl_density_sample`; `out` is null or writable for `cap` doubles;
/// `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_density_correction_fit(
    samples: *const SlDensitySample,
    n: usize,
    ridge: f64,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_density_correction_fit";
        if samples.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: null samples pointer"),
            );
        }
        if let Err(status) = check_nonnegative(WHO, "ridge", ridge) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is readable for `n` elements.
        let rows = unsafe { borrow_density_samples(samples, n) };
        let Some(correction) =
            DensityCorrection::<DENSITY_CORRECTION_COEFFICIENTS>::fit_ridge(&rows, ridge)
        else {
            return fail(
                SL_ERR_COMPUTE,
                format!("{WHO}: the normal equations over {n} samples are singular"),
            );
        };
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(correction.coefficients(), out, cap, out_len) }
    })
}

/// Bin the samples into an `n_lat × n_lst` field of the mean `log_ratio`, row-major
/// `lat_bin * n_lst + lst_bin` with empty cells zero — a density-field snapshot for POD.
/// Two-call protocol; the result is always `n_lat * n_lst` long.
///
/// # Safety
/// `samples` is readable for `n` `sl_density_sample`; `out` is null or writable for `cap` doubles;
/// `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_density_grid_snapshot(
    samples: *const SlDensitySample,
    n: usize,
    n_lat: usize,
    n_lst: usize,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_density_grid_snapshot";
        if samples.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: null samples pointer"),
            );
        }
        if let Err(status) =
            check_nonzero(WHO, "n_lat", n_lat).and_then(|()| check_nonzero(WHO, "n_lst", n_lst))
        {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is readable for `n` elements.
        let rows = unsafe { borrow_density_samples(samples, n) };
        let grid = grid_samples(&rows, n_lat, n_lst);
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&grid, out, cap, out_len) }
    })
}

/// The two lengths a swath is measured against must be positive, and the angle merely finite: a
/// look angle past the horizon is a domain answer the geometry itself gives, not an input error.
fn check_swath_inputs(
    who: &str,
    earth_radius_m: f64,
    altitude_m: f64,
    look_angle_rad: f64,
) -> Result<(), i32> {
    check_positive(who, "earth_radius_m", earth_radius_m, SL_ERR_DOMAIN)?;
    check_positive(who, "altitude_m", altitude_m, SL_ERR_DOMAIN)?;
    check_finite(who, "look_angle_rad", look_angle_rad)
}

/// The swath geometry one look angle cuts out of a spherical Earth.
///
/// A look angle past the horizon has no ground point at all, which is `SL_ERR_DOMAIN` rather than
/// a NaN: the answer does not exist, and saying so is not the same as computing one badly.
///
/// # Safety
/// `out` is writable for one `sl_swath`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_swath_geometry(
    earth_radius_m: f64,
    altitude_m: f64,
    look_angle_rad: f64,
    out: *mut SlSwath,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_swath_geometry";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out pointer"));
        }
        if let Err(status) = check_swath_inputs(WHO, earth_radius_m, altitude_m, look_angle_rad) {
            return status;
        }
        let Some(g) = ev_swath_geometry(earth_radius_m, altitude_m, look_angle_rad) else {
            return fail(
                SL_ERR_DOMAIN,
                format!("{WHO}: a look angle of {look_angle_rad} rad misses the Earth"),
            );
        };
        // SAFETY: checked non-null.
        unsafe {
            *out = SlSwath {
                earth_central_angle_rad: g.earth_central_angle,
                elevation_rad: g.elevation,
                ground_range_m: g.ground_range,
                slant_range_m: g.slant_range,
            }
        };
        SL_OK
    })
}

// ── Numerical propagation: the force models the dense handle can be built over ─────────────────

/// `sl_eop`: the Earth-orientation quantities one arc is rotated with, held fixed over the arc.
///
/// There is no length-of-day field, and its absence is deliberate: the GCRF→ITRF rotation the
/// propagators use is independent of LOD, which only scales the angular velocity they discard. A
/// field the library reads and ignores is worse than no field, because a caller who fills it in
/// believes something is happening.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlEop {
    /// The arc's start on TT, nanoseconds since J2000.
    pub epoch_tt_ns: i64,
    /// The same instant on UT1.
    pub epoch_ut1_ns: i64,
    /// Polar motion, radians.
    pub xp_rad: f64,
    /// The other component.
    pub yp_rad: f64,
    /// Celestial-pole offset, radians.
    pub dx_rad: f64,
    /// The other component.
    pub dy_rad: f64,
}

/// `sl_eop_sample`: one row of an EOP series, interpolated linearly between rows.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlEopSample {
    /// Seconds from the arc's epoch.
    pub t_seconds: f64,
    /// Polar motion, radians.
    pub xp_rad: f64,
    /// The other component.
    pub yp_rad: f64,
    /// UT1 − UTC, seconds.
    pub dut1_seconds: f64,
    /// Celestial-pole offset, radians.
    pub dx_rad: f64,
    /// The other component.
    pub dy_rad: f64,
}

/// `sl_exponential_atmosphere`: a scale-height atmosphere and the ballistic coefficient through it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlExponentialAtmosphere {
    /// `m / (Cd A)`, kg/m².
    pub ballistic_coefficient: f64,
    /// Density at the reference altitude, kg/m³.
    pub reference_density: f64,
    /// Where that density is quoted, metres above the surface.
    pub reference_altitude_m: f64,
    /// The e-folding height, metres.
    pub scale_height_m: f64,
    /// The spherical Earth the altitude is measured from, metres.
    pub earth_radius_m: f64,
}

/// `sl_nrlmsise_drag`: drag through NRLMSISE-00 rather than through a scale height.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlNrlmsiseDrag {
    /// Day of year at the arc's epoch.
    pub doy: i32,
    /// UT seconds into that day.
    pub seconds_of_day: f64,
    /// `m / (Cd A)`, kg/m².
    pub ballistic_coefficient: f64,
    /// Daily 10.7 cm solar flux, sfu.
    pub f107: f64,
    /// Its 81-day mean, sfu.
    pub f107a: f64,
    /// The daily planetary amplitude.
    pub ap: f64,
    /// Gravitational parameter, m³/s².
    pub mu: f64,
}

/// An arc has to end somewhere, and a zero-length one is not an arc.
fn check_t1(who: &str, t1_seconds: f64) -> Result<(), i32> {
    if !t1_seconds.is_finite() || t1_seconds == 0.0 {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: t1_seconds is {t1_seconds}, and must be finite and non-zero"),
        ));
    }
    Ok(())
}

/// The two checks every dense constructor makes on the arc it is asked for.
fn check_arc(who: &str, t1_seconds: f64, tolerance: f64) -> Result<(), i32> {
    check_t1(who, t1_seconds)?;
    if !(tolerance.is_finite() && tolerance > 0.0) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: tolerance is {tolerance}, and must be finite and positive"),
        ));
    }
    Ok(())
}

/// The embedded field at `degree`, or the refusal that says why it is not there.
fn embedded_field(who: &str, degree: u32) -> Result<GeoPotential, i32> {
    GeoPotential::embedded(degree as usize).ok_or_else(|| {
        fail(
            SL_ERR_DOMAIN,
            format!("{who}: degree {degree} exceeds the embedded model (max {MODEL_MAX_DEGREE})"),
        )
    })
}

/// What an EOP series has to be before it can be interpolated: strictly increasing in `t_seconds`,
/// and finite in every field of every row. A safe helper rather than an inline check because
/// `cargo mutants` generates nothing for the body of an `unsafe fn`, so a condition written inside
/// one of the entry points is outside the seal.
fn check_eop_series(who: &str, rows: &[SlEopSample]) -> Result<(), i32> {
    for (i, row) in rows.iter().enumerate() {
        for (name, value) in [
            ("t_seconds", row.t_seconds),
            ("xp_rad", row.xp_rad),
            ("yp_rad", row.yp_rad),
            ("dut1_seconds", row.dut1_seconds),
            ("dx_rad", row.dx_rad),
            ("dy_rad", row.dy_rad),
        ] {
            if !value.is_finite() {
                return Err(fail(
                    SL_ERR_INVALID_ARGUMENT,
                    format!("{who}: row {i} has {name} = {value}, and must be finite"),
                ));
            }
        }
    }
    if rows.windows(2).any(|w| w[0].t_seconds >= w[1].t_seconds) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: the series is not strictly increasing in t_seconds"),
        ));
    }
    Ok(())
}

/// The interpolated GCRF→ITRF provider one `sl_eop` describes, over `[0, span]`.
fn cio_over(eop: SlEop, span: f64) -> CioInterpolator {
    CioInterpolator::new(
        Epoch::<Ut1>::from_ns_since_j2000(i128::from(eop.epoch_ut1_ns)),
        Epoch::<Tt>::from_ns_since_j2000(i128::from(eop.epoch_tt_ns)),
        span,
        Angle::from_radians(eop.xp_rad),
        Angle::from_radians(eop.yp_rad),
        Angle::from_radians(eop.dx_rad),
        Angle::from_radians(eop.dy_rad),
        0.0,
    )
}

/// Hand a DOP853 dense integration's result to the caller as a handle. Every constructor below
/// differs only in the force model it composes, so the failure message is written here once.
///
/// # Safety
/// `out` is writable for one pointer.
unsafe fn deliver_dop853<E: core::fmt::Debug>(
    who: &str,
    integrate: impl FnOnce() -> Result<Trajectory<6, Dop853Dense<6>>, E>,
    out: *mut *mut SlEphemeris,
) -> i32 {
    match integrate() {
        Ok(traj) => {
            // SAFETY: the caller states `out` is writable for one pointer.
            unsafe { deliver_ephemeris(EphemerisKind::Dop853(Box::new(traj)), out) }
        }
        Err(e) => fail(
            SL_ERR_COMPUTE,
            format!("{who}: dense propagation failed: {e:?}"),
        ),
    }
}

/// The linear interpolation an EOP series is read with, saturating at both ends rather than
/// extrapolating a quantity that is measured and not modelled.
fn lerp_at(samples: &[SlEopSample], pick: impl Fn(&SlEopSample) -> f64, t: f64) -> f64 {
    let n = samples.len();
    if t <= samples[0].t_seconds {
        return pick(&samples[0]);
    }
    if t >= samples[n - 1].t_seconds {
        return pick(&samples[n - 1]);
    }
    let i = samples.partition_point(|s| s.t_seconds <= t) - 1;
    let frac = (t - samples[i].t_seconds) / (samples[i + 1].t_seconds - samples[i].t_seconds);
    pick(&samples[i]) + frac * (pick(&samples[i + 1]) - pick(&samples[i]))
}

/// Propagate under point-mass gravity plus J2 with DOP853, keeping the dense interpolant.
///
/// The same forces as `sl_ephemeris_j2` under a different integrator; see the header for which to
/// reach for.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `out` is writable for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_j2_dense(
    r#in: *const SlState,
    t1_seconds: f64,
    mu: f64,
    j2: f64,
    earth_radius_m: f64,
    tolerance: f64,
    out: *mut *mut SlEphemeris,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_ephemeris_j2_dense";
        if r#in.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_arc(WHO, t1_seconds, tolerance) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let y0 = to_six(unsafe { *r#in });
        let sys = Plain(Newtonian((
            PointMass { mu },
            J2 {
                mu,
                re: earth_radius_m,
                j2,
            },
        )));
        // SAFETY: `out` was checked non-null above.
        unsafe {
            deliver_dop853(
                WHO,
                || {
                    integrate_dop853(
                        &sys,
                        0.0,
                        &y0,
                        t1_seconds,
                        initial_step(t1_seconds),
                        Controls::new(Tolerance::new(tolerance, tolerance), 1e-3),
                    )
                },
                out,
            )
        }
    })
}

/// Propagate under the Earth-fixed geopotential plus drag through an exponential atmosphere.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `atmosphere` at one initialised
/// `sl_exponential_atmosphere`; `out` is writable for one pointer.
#[allow(clippy::too_many_arguments)] // the arc, the field, the rotation phase and the atmosphere
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_drag(
    r#in: *const SlState,
    t1_seconds: f64,
    degree: u32,
    theta0_rad: f64,
    atmosphere: *const SlExponentialAtmosphere,
    tolerance: f64,
    out: *mut *mut SlEphemeris,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_ephemeris_drag";
        if r#in.is_null() || atmosphere.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_arc(WHO, t1_seconds, tolerance) {
            return status;
        }
        let geo = match embedded_field(WHO, degree) {
            Ok(geo) => geo,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states both are initialised.
        let (y0, a) = unsafe { (to_six(*r#in), *atmosphere) };
        if let Err(status) = check_positive(WHO, "scale_height_m", a.scale_height_m, SL_ERR_DOMAIN)
        {
            return status;
        }
        let sys = Plain(Newtonian((
            EarthFixed {
                inner: geo,
                omega: spacelab_frames::EARTH_ROTATION_RATE,
                theta0: theta0_rad,
            },
            Drag {
                ballistic_coefficient: a.ballistic_coefficient,
                earth_rotation_rate: spacelab_frames::EARTH_ROTATION_RATE,
                atmosphere: ExponentialAtmosphere {
                    reference_density: a.reference_density,
                    reference_altitude: a.reference_altitude_m,
                    scale_height: a.scale_height_m,
                    earth_radius: a.earth_radius_m,
                },
                wind: NoWind,
            },
        )));
        // SAFETY: `out` was checked non-null above.
        unsafe {
            deliver_dop853(
                WHO,
                || {
                    integrate_dop853(
                        &sys,
                        0.0,
                        &y0,
                        t1_seconds,
                        initial_step(t1_seconds),
                        Controls::new(Tolerance::new(tolerance, tolerance), 1e-3),
                    )
                },
                out,
            )
        }
    })
}

/// Propagate under point-mass gravity plus drag through NRLMSISE-00, whose density is read at the
/// geodetic point the state rotates to at each stage.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `eop` at one `sl_eop`; `drag` at one
/// `sl_nrlmsise_drag`; `out` is writable for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_drag_nrlmsise00(
    r#in: *const SlState,
    t1_seconds: f64,
    eop: *const SlEop,
    drag: *const SlNrlmsiseDrag,
    tolerance: f64,
    out: *mut *mut SlEphemeris,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_ephemeris_drag_nrlmsise00";
        if r#in.is_null() || eop.is_null() || drag.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_arc(WHO, t1_seconds, tolerance) {
            return status;
        }
        // SAFETY: checked non-null; the caller states all three are initialised.
        let (y0, eop, d) = unsafe { (to_six(*r#in), *eop, *drag) };
        let cio = cio_over(eop, t1_seconds);
        let geodetic = move |t: f64, rf: [f64; 3]| {
            let m = cio.rotation_rows(t);
            let r_itrf = Vec3::new(
                m[0][0] * rf[0] + m[0][1] * rf[1] + m[0][2] * rf[2],
                m[1][0] * rf[0] + m[1][1] * rf[1] + m[1][2] * rf[2],
                m[2][0] * rf[0] + m[2][1] * rf[1] + m[2][2] * rf[2],
            );
            match earth_itrf_to_geodetic(Position::new(r_itrf)) {
                Ok(g) => (
                    g.height.metres() / 1000.0,
                    g.latitude.degrees(),
                    g.longitude.degrees(),
                ),
                Err(_) => (0.0, 0.0, 0.0),
            }
        };
        let sys = Plain(Newtonian((
            PointMass { mu: d.mu },
            Drag {
                ballistic_coefficient: d.ballistic_coefficient,
                earth_rotation_rate: spacelab_frames::EARTH_ROTATION_RATE,
                atmosphere: Nrlmsise00Atmosphere {
                    doy0: d.doy,
                    sec0: d.seconds_of_day,
                    f107: d.f107,
                    f107a: d.f107a,
                    ap: d.ap,
                    geodetic,
                },
                wind: NoWind,
            },
        )));
        // SAFETY: `out` was checked non-null above.
        unsafe {
            deliver_dop853(
                WHO,
                || {
                    integrate_dop853(
                        &sys,
                        0.0,
                        &y0,
                        t1_seconds,
                        initial_step(t1_seconds),
                        Controls::new(Tolerance::new(tolerance, tolerance), 1e-3),
                    )
                },
                out,
            )
        }
    })
}

/// Propagate under the geopotential evaluated in the true ITRF, rotated by the interpolated CIO
/// chain rather than by a uniform sidereal angle.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `eop` at one `sl_eop`; `out` is writable for one
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_geopotential_itrf(
    r#in: *const SlState,
    t1_seconds: f64,
    degree: u32,
    eop: *const SlEop,
    tolerance: f64,
    out: *mut *mut SlEphemeris,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_ephemeris_geopotential_itrf";
        if r#in.is_null() || eop.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_arc(WHO, t1_seconds, tolerance) {
            return status;
        }
        let geo = match embedded_field(WHO, degree) {
            Ok(geo) => geo,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states both are initialised.
        let (y0, eop) = unsafe { (to_six(*r#in), *eop) };
        let cio = cio_over(eop, t1_seconds);
        let rotate = move |t: f64| cio.rotation_rows(t);
        let sys = Plain(Newtonian(BodyOriented { inner: geo, rotate }));
        // SAFETY: `out` was checked non-null above.
        unsafe {
            deliver_dop853(
                WHO,
                || {
                    integrate_dop853(
                        &sys,
                        0.0,
                        &y0,
                        t1_seconds,
                        initial_step(t1_seconds),
                        Controls::new(Tolerance::new(tolerance, tolerance), 1e-3),
                    )
                },
                out,
            )
        }
    })
}

/// The same, plus the full IERS 2010 degree-2 solid Earth tide: Step 1 and the frequency-dependent
/// Step 2, with the solid and ocean pole tides on the ΔC21/ΔS21 band.
///
/// Both the field and the tide need the frame, so one interpolated rotation is shared between them
/// rather than gridded twice.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `eop` at one `sl_eop`; `out` is writable for one
/// pointer.
#[allow(clippy::too_many_arguments)] // the arc, the field, the frame, the epoch and two attractors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_geopotential_tide(
    r#in: *const SlState,
    t1_seconds: f64,
    degree: u32,
    eop: *const SlEop,
    epoch_j2000_seconds: f64,
    mu_sun: f64,
    mu_moon: f64,
    tolerance: f64,
    out: *mut *mut SlEphemeris,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_ephemeris_geopotential_tide";
        if r#in.is_null() || eop.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_arc(WHO, t1_seconds, tolerance) {
            return status;
        }
        let geo = match embedded_field(WHO, degree) {
            Ok(geo) => geo,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states both are initialised.
        let (y0, eop) = unsafe { (to_six(*r#in), *eop) };
        let cio = Rc::new(cio_over(eop, t1_seconds));
        let cio_tide = Rc::clone(&cio);
        let rotate = move |t: f64| cio.rotation_rows(t);
        let rotate_tide = move |t: f64| cio_tide.rotation_rows(t);
        let (epoch_tt_ns, epoch_ut1_ns) =
            (i128::from(eop.epoch_tt_ns), i128::from(eop.epoch_ut1_ns));
        let tide = SolidEarthTide {
            mu_earth: MODEL_GM,
            re_earth: MODEL_RE,
            mu_sun,
            mu_moon,
            sun: AnalyticSun {
                epoch_j2000_seconds,
            },
            moon: AnalyticMoon {
                epoch_j2000_seconds,
            },
            rotate: rotate_tide,
            step2: move |t: f64| {
                let dt_ns = (t * 1e9).round() as i128;
                let tt = Epoch::<Tt>::from_ns_since_j2000(epoch_tt_ns + dt_ns);
                let ut1 = Epoch::<Ut1>::from_ns_since_j2000(epoch_ut1_ns + dt_ns);
                let tc = tt.julian_centuries();
                let delaunay = [
                    spacelab_fa::fundamental::fal03(tc),
                    spacelab_fa::fundamental::falp03(tc),
                    spacelab_fa::fundamental::faf03(tc),
                    spacelab_fa::fundamental::fad03(tc),
                    spacelab_fa::fundamental::faom03(tc),
                ];
                let t_j2000 = (epoch_tt_ns + dt_ns) as f64 / 1e9;
                let mut s = step2_corrections(spacelab_fa::gmst82(ut1).radians(), delaunay);
                let pole = MeanPole::Secular2018;
                let sp = pole_tide_corrections(eop.xp_rad, eop.yp_rad, t_j2000, pole);
                let op = ocean_pole_tide_corrections(eop.xp_rad, eop.yp_rad, t_j2000, pole);
                s[1] += sp[0] + op[0]; // ΔC̄21
                s[2] += sp[1] + op[1]; // ΔS̄21
                s
            },
        };
        let sys = Plain(Newtonian((BodyOriented { inner: geo, rotate }, tide)));
        // SAFETY: `out` was checked non-null above.
        unsafe {
            deliver_dop853(
                WHO,
                || {
                    integrate_dop853(
                        &sys,
                        0.0,
                        &y0,
                        t1_seconds,
                        initial_step(t1_seconds),
                        Controls::new(Tolerance::new(tolerance, tolerance), 1e-3),
                    )
                },
                out,
            )
        }
    })
}

/// Propagate under point-mass gravity plus J2 evaluated in the true ITRF.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `eop` at one `sl_eop`; `out` is writable for one
/// pointer.
#[allow(clippy::too_many_arguments)] // the arc, three field constants, the frame and the tolerance
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_j2_itrf(
    r#in: *const SlState,
    t1_seconds: f64,
    mu: f64,
    j2: f64,
    earth_radius_m: f64,
    eop: *const SlEop,
    tolerance: f64,
    out: *mut *mut SlEphemeris,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_ephemeris_j2_itrf";
        if r#in.is_null() || eop.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_arc(WHO, t1_seconds, tolerance) {
            return status;
        }
        // SAFETY: checked non-null; the caller states both are initialised.
        let (y0, eop) = unsafe { (to_six(*r#in), *eop) };
        let cio = cio_over(eop, t1_seconds);
        let rotate = move |t: f64| cio.rotation_rows(t);
        let sys = Plain(Newtonian((
            PointMass { mu },
            BodyOriented {
                inner: J2 {
                    mu,
                    re: earth_radius_m,
                    j2,
                },
                rotate,
            },
        )));
        // SAFETY: `out` was checked non-null above.
        unsafe {
            deliver_dop853(
                WHO,
                || {
                    integrate_dop853(
                        &sys,
                        0.0,
                        &y0,
                        t1_seconds,
                        initial_step(t1_seconds),
                        Controls::new(Tolerance::new(tolerance, tolerance), 1e-3),
                    )
                },
                out,
            )
        }
    })
}

/// The same, over a *series* of Earth-orientation rows rather than one fixed set: the rotation is
/// rebuilt from the interpolated EOP at every stage, which is what a multi-day arc needs.
///
/// The series must be sorted by `t_seconds` and hold at least two rows; a query outside it
/// saturates at the nearest end rather than extrapolating a measured quantity.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `series` is readable for `n` `sl_eop_sample`; `out`
/// is writable for one pointer.
#[allow(clippy::too_many_arguments)] // the arc, three field constants, the series and its epoch
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_j2_itrf_series(
    r#in: *const SlState,
    t1_seconds: f64,
    mu: f64,
    j2: f64,
    earth_radius_m: f64,
    epoch_utc_ns: i64,
    tt_minus_utc_s: f64,
    series: *const SlEopSample,
    n: usize,
    tolerance: f64,
    out: *mut *mut SlEphemeris,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_ephemeris_j2_itrf_series";
        if r#in.is_null() || series.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_arc(WHO, t1_seconds, tolerance) {
            return status;
        }
        if let Err(status) = check_at_least(WHO, "n", n, 2) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is readable for `n` rows.
        let rows = unsafe { core::slice::from_raw_parts(series, n) }.to_vec();
        if let Err(status) = check_eop_series(WHO, &rows) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let y0 = to_six(unsafe { *r#in });
        let epoch_utc_ns = i128::from(epoch_utc_ns);
        let rotate = move |t: f64| {
            let dut1 = lerp_at(&rows, |s| s.dut1_seconds, t);
            let tt = Epoch::<Tt>::from_ns_since_j2000(
                epoch_utc_ns + ((t + tt_minus_utc_s) * 1e9).round() as i128,
            );
            let ut1 = Epoch::<Ut1>::from_ns_since_j2000(
                epoch_utc_ns + ((t + dut1) * 1e9).round() as i128,
            );
            frames_gcrf_to_itrf(
                ut1,
                tt,
                Angle::from_radians(lerp_at(&rows, |s| s.xp_rad, t)),
                Angle::from_radians(lerp_at(&rows, |s| s.yp_rad, t)),
                Angle::from_radians(lerp_at(&rows, |s| s.dx_rad, t)),
                Angle::from_radians(lerp_at(&rows, |s| s.dy_rad, t)),
                0.0,
            )
            .rotation
            .matrix()
            .rows
        };
        let sys = Plain(Newtonian((
            PointMass { mu },
            BodyOriented {
                inner: J2 {
                    mu,
                    re: earth_radius_m,
                    j2,
                },
                rotate,
            },
        )));
        // SAFETY: `out` was checked non-null above.
        unsafe {
            deliver_dop853(
                WHO,
                || {
                    integrate_dop853(
                        &sys,
                        0.0,
                        &y0,
                        t1_seconds,
                        initial_step(t1_seconds),
                        Controls::new(Tolerance::new(tolerance, tolerance), 1e-3),
                    )
                },
                out,
            )
        }
    })
}

/// Propagate under the Earth-fixed geopotential plus Sun and Moon third-body attraction and solar
/// radiation pressure, with the analytic ephemerides.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `out` is writable for one pointer.
#[allow(clippy::too_many_arguments)] // the arc, the field, the epoch, the SRP scale and the phase
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_luni_solar(
    r#in: *const SlState,
    t1_seconds: f64,
    degree: u32,
    epoch_j2000_seconds: f64,
    srp_reference_acceleration: f64,
    theta0_rad: f64,
    tolerance: f64,
    out: *mut *mut SlEphemeris,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_ephemeris_luni_solar";
        if r#in.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_arc(WHO, t1_seconds, tolerance) {
            return status;
        }
        let geo = match embedded_field(WHO, degree) {
            Ok(geo) => geo,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states it is initialised.
        let y0 = to_six(unsafe { *r#in });
        let sun = AnalyticSun {
            epoch_j2000_seconds,
        };
        let moon = AnalyticMoon {
            epoch_j2000_seconds,
        };
        let sys = Plain(Newtonian((
            EarthFixed {
                inner: geo,
                omega: spacelab_frames::EARTH_ROTATION_RATE,
                theta0: theta0_rad,
            },
            ThirdBody {
                mu: MU_SUN,
                ephemeris: sun,
            },
            ThirdBody {
                mu: MU_MOON,
                ephemeris: moon,
            },
            SolarRadiationPressure {
                reference_acceleration: srp_reference_acceleration,
                astronomical_unit: ASTRONOMICAL_UNIT,
                earth_radius: RE_EARTH,
                sun,
            },
        )));
        // SAFETY: `out` was checked non-null above.
        unsafe {
            deliver_dop853(
                WHO,
                || {
                    integrate_dop853(
                        &sys,
                        0.0,
                        &y0,
                        t1_seconds,
                        initial_step(t1_seconds),
                        Controls::new(Tolerance::new(tolerance, tolerance), 1e-3),
                    )
                },
                out,
            )
        }
    })
}

/// Propagate under point-mass gravity plus the three IERS 2010 relativistic corrections:
/// Schwarzschild, Lense–Thirring and de Sitter.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `out` is writable for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ephemeris_relativistic(
    r#in: *const SlState,
    t1_seconds: f64,
    epoch_j2000_seconds: f64,
    mu: f64,
    mu_sun: f64,
    tolerance: f64,
    out: *mut *mut SlEphemeris,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_ephemeris_relativistic";
        if r#in.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_arc(WHO, t1_seconds, tolerance) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let y0 = to_six(unsafe { *r#in });
        // The de Sitter term needs the Sun's velocity, and the analytic ephemeris gives position
        // only. A 60 s central difference is far finer than a ~4e-11 m/s² term requires.
        let sun_pv = move |t: f64| {
            let sun = AnalyticSun {
                epoch_j2000_seconds,
            };
            let dt = 60.0;
            let (pm, pp) = (sun.position(t - dt), sun.position(t + dt));
            (
                sun.position(t),
                [
                    (pp[0] - pm[0]) / (2.0 * dt),
                    (pp[1] - pm[1]) / (2.0 * dt),
                    (pp[2] - pm[2]) / (2.0 * dt),
                ],
            )
        };
        let sys = Plain(Newtonian((
            PointMass { mu },
            Schwarzschild { mu },
            LenseThirring {
                mu,
                j: [0.0, 0.0, EARTH_ANGULAR_MOMENTUM_PER_MASS],
            },
            DeSitter { mu_sun, sun_pv },
        )));
        // SAFETY: `out` was checked non-null above.
        unsafe {
            deliver_dop853(
                WHO,
                || {
                    integrate_dop853(
                        &sys,
                        0.0,
                        &y0,
                        t1_seconds,
                        initial_step(t1_seconds),
                        Controls::new(Tolerance::new(tolerance, tolerance), 1e-3),
                    )
                },
                out,
            )
        }
    })
}

/// Fixed-step 8th-order symplectic (Yoshida) propagation under point-mass gravity plus J2, in `f64`
/// or in double-double — the arithmetic itself is the parameter, which is the point of the call.
///
/// # Safety
/// `in` points at one initialised `sl_state`; `out` is writable for one.
#[allow(clippy::too_many_arguments)] // three field constants, the step, the count and the arithmetic
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_propagate_j2_precision(
    r#in: *const SlState,
    mu: f64,
    j2: f64,
    earth_radius_m: f64,
    dt_seconds: f64,
    steps: u64,
    precision: i32,
    out: *mut SlState,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_propagate_j2_precision";
        if r#in.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_finite(WHO, "dt_seconds", dt_seconds) {
            return status;
        }
        let steps = match check_steps(WHO, steps) {
            Ok(steps) => steps,
            Err(status) => return status,
        };
        let precision = match precision_of(WHO, precision) {
            Ok(precision) => precision,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states it is initialised.
        let state = unsafe { *r#in };
        let (r0, v0) = (to_three(state.position), to_three(state.velocity));
        let force = (
            PointMass { mu },
            J2 {
                mu,
                re: earth_radius_m,
                j2,
            },
        );
        let (r, v) = match precision {
            Precision::Double => {
                let accel = |x: &[f64; 3]| force.acceleration::<f64>(0.0, x, &[0.0; 3]);
                let (mut x, mut v) = (r0, v0);
                yoshida8_scalar(&accel, &mut x, &mut v, dt_seconds, steps);
                (x, v)
            }
            Precision::DoubleDouble => {
                let zero = [DoubleDouble::from_f64(0.0); 3];
                let accel = |x: &[DoubleDouble; 3]| {
                    force.acceleration::<DoubleDouble>(DoubleDouble::from_f64(0.0), x, &zero)
                };
                let mut x = r0.map(DoubleDouble::from_f64);
                let mut v = v0.map(DoubleDouble::from_f64);
                yoshida8_scalar(&accel, &mut x, &mut v, dt_seconds, steps);
                (x.map(|c| c.value()), v.map(|c| c.value()))
            }
        };
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe { *out = state_of(r, v) };
        SL_OK
    })
}

// ── Phase 8: SAR, InSAR, TomoSAR and polarimetry ───────────────────────────────────────────────

/// `sl_complex`: one complex sample, laid out as C's `double _Complex` is — real part first.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlComplex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

/// `sl_csa_params`: the acquisition a chirp-scaling focus is run against.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlCsaParams {
    /// Pulse repetition frequency, hertz.
    pub prf: f64,
    /// Platform velocity, m/s.
    pub velocity: f64,
    /// Carrier wavelength, metres.
    pub wavelength: f64,
    /// Reference slant range the scaling is built at, metres.
    pub r_ref: f64,
    /// Range sample rate, hertz.
    pub range_sample_rate: f64,
    /// Range chirp rate, Hz/s.
    pub range_fm_rate: f64,
}

/// `sl_rda_geometry`: the acquisition the range-Doppler rates are derived from.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlRdaGeometry {
    /// Pulse repetition frequency, hertz.
    pub prf: f64,
    /// Platform velocity, m/s.
    pub velocity: f64,
    /// Carrier wavelength, metres.
    pub wavelength: f64,
    /// Slant range at the near edge, metres.
    pub r0_slant: f64,
    /// Range sample spacing, metres.
    pub range_spacing: f64,
}

/// `sl_tomo_params`: the tomographic geometry an elevation spectrum is formed over.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlTomoParams {
    /// Carrier wavelength, metres.
    pub wavelength: f64,
    /// Slant range to the resolution cell, metres.
    pub slant_range: f64,
    /// Diagonal loading, as a fraction of the covariance trace. Capon only.
    pub loading: f64,
    /// L1 regularisation weight. SL1MMER only.
    pub lambda: f64,
    /// Iteration cap. SL1MMER only.
    pub max_iter: usize,
    /// Relative amplitude below which a scatterer is dropped. SL1MMER only.
    pub threshold_rel: f64,
}

/// `sl_scatterer`: one elevation and the complex reflectivity SL1MMER put there.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlScatterer {
    /// Elevation along the tomographic aperture, metres.
    pub elevation_m: f64,
    /// Complex reflectivity.
    pub reflectivity: SlComplex,
}

/// `sl_polarimetric_sample`: one look's three-component Pauli scattering vector.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlPolarimetricSample {
    /// Co-polar HH.
    pub s_hh: SlComplex,
    /// Cross-polar HV.
    pub s_hv: SlComplex,
    /// Co-polar VV.
    pub s_vv: SlComplex,
}

/// `sl_cloude_pottier`: the three parameters of the H/A/alpha decomposition.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlCloudePottier {
    /// Scattering entropy, in `[0, 1]`.
    pub entropy: f64,
    /// Anisotropy, in `[0, 1]`.
    pub anisotropy: f64,
    /// Mean alpha angle, radians.
    pub alpha_rad: f64,
}

/// `sl_deformation`: what an SBAS inversion resolves a stack into.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlDeformation {
    /// Line-of-sight velocity, m/s.
    pub velocity_ms: f64,
    /// Residual topographic height, metres.
    pub height_m: f64,
}

/// How many complex entries a 3x3 coherency matrix holds.
const COHERENCY_ENTRIES: usize = 9;

/// The crate's own acquisition record, from the C one.
fn rda_of(g: SlRdaGeometry) -> RdaParams {
    RdaParams {
        prf: g.prf,
        velocity: g.velocity,
        wavelength: g.wavelength,
        r0_slant: g.r0_slant,
        range_spacing: g.range_spacing,
    }
}

fn to_complex(c: SlComplex) -> SigComplex {
    SigComplex::new(c.re, c.im)
}

fn from_complex(c: SigComplex) -> SlComplex {
    SlComplex { re: c.re, im: c.im }
}

/// Borrow a caller's complex array as the crate's own.
///
/// # Safety
/// `p` is readable for `n` `sl_complex`.
unsafe fn borrow_complex(p: *const SlComplex, n: usize) -> Vec<SigComplex> {
    // SAFETY: the caller states `p` is readable for `n` elements.
    unsafe { core::slice::from_raw_parts(p, n) }
        .iter()
        .map(|&c| to_complex(c))
        .collect()
}

/// Deliver a complex answer through the two-call protocol.
///
/// # Safety
/// `out` is null or writable for `cap` `sl_complex`; `out_len` is writable for one `size_t`.
unsafe fn deliver_complex(
    values: &[SigComplex],
    out: *mut SlComplex,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    let converted: Vec<SlComplex> = values.iter().map(|&c| from_complex(c)).collect();
    // SAFETY: forwarding the caller's own contract.
    unsafe { deliver(&converted, out, cap, out_len) }
}

/// A linear frequency-modulated chirp of `n` samples. Two-call protocol.
///
/// # Safety
/// `out` is null or writable for `cap` `sl_complex`; `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_lfm_chirp(
    n: usize,
    chirp_rate: f64,
    sample_rate: f64,
    out: *mut SlComplex,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_lfm_chirp";
        if let Err(status) = check_nonzero(WHO, "n", n)
            .and_then(|()| check_positive(WHO, "sample_rate", sample_rate, SL_ERR_DOMAIN))
        {
            return status;
        }
        let chirp = sig_lfm_chirp(n, chirp_rate, sample_rate);
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver_complex(&chirp, out, cap, out_len) }
    })
}

/// Correlate a signal against a replica — the compression step of every SAR pipeline. Two-call.
///
/// # Safety
/// `signal` is readable for `signal_len` and `replica` for `replica_len`; `out` is null or writable
/// for `cap`; `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_matched_filter(
    signal: *const SlComplex,
    signal_len: usize,
    replica: *const SlComplex,
    replica_len: usize,
    out: *mut SlComplex,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_matched_filter";
        if signal.is_null() || replica.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_nonzero(WHO, "signal_len", signal_len)
            .and_then(|()| check_nonzero(WHO, "replica_len", replica_len))
        {
            return status;
        }
        // SAFETY: checked non-null; the caller states both are readable for their lengths.
        let (s, r) = unsafe {
            (
                borrow_complex(signal, signal_len),
                borrow_complex(replica, replica_len),
            )
        };
        let compressed = sig_matched_filter(&s, &r);
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver_complex(&compressed, out, cap, out_len) }
    })
}

/// The range cell migration `l^2 r0 f_eta^2 / (8 V^2)`, in metres, at one azimuth frequency. Total.
///
/// # Safety
/// `geometry` is null, or points at one initialised `sl_rda_geometry`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_range_migration(f_eta: f64, geometry: *const SlRdaGeometry) -> f64 {
    guard_f64(|| {
        if geometry.is_null() {
            let _ = fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_range_migration: null geometry pointer",
            );
            return f64::NAN;
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let g = unsafe { *geometry };
        sig_range_migration(f_eta, &rda_of(g))
    })
}

/// The secondary range compression FM rate at one azimuth frequency.
///
/// A geometry whose azimuth frequency is past the Doppler bandwidth has no SRC rate at all, which
/// is `SL_ERR_DOMAIN` rather than a NaN.
///
/// # Safety
/// `geometry` points at one initialised `sl_rda_geometry`; `out` is writable for one double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_src_fm_rate(
    f_eta: f64,
    geometry: *const SlRdaGeometry,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_src_fm_rate";
        if geometry.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let g = unsafe { *geometry };
        let Some(rate) = sig_src_fm_rate(f_eta, &rda_of(g)) else {
            return fail(
                SL_ERR_DOMAIN,
                format!("{WHO}: no SRC rate exists at f_eta = {f_eta} for this geometry"),
            );
        };
        // SAFETY: checked non-null.
        unsafe { *out = rate };
        SL_OK
    })
}

/// The azimuth FM rate of the acquisition, Hz/s. Total.
///
/// # Safety
/// `geometry` is null, or points at one initialised `sl_rda_geometry`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_azimuth_fm_rate(geometry: *const SlRdaGeometry) -> f64 {
    guard_f64(|| {
        if geometry.is_null() {
            let _ = fail(
                SL_ERR_INVALID_ARGUMENT,
                "sl_azimuth_fm_rate: null geometry pointer",
            );
            return f64::NAN;
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let g = unsafe { *geometry };
        sig_azimuth_fm_rate(&rda_of(g))
    })
}

/// A chirp-scaling aperture is a raster whose two sides are also radix-2 FFT lengths.
///
/// The transform asserts the power of two rather than returning, and a panic unwound into a C
/// caller is not an answer, so it is decided here, where there is still a status to report it in.
fn check_fft_aperture(who: &str, na: usize, nr: usize) -> Result<usize, i32> {
    let n = check_raster(who, na, nr)?;
    if !na.is_power_of_two() || !nr.is_power_of_two() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: the aperture is {na} x {nr}, and both must be powers of two"),
        ));
    }
    Ok(n)
}

/// Focus a raw SAR aperture with the chirp-scaling algorithm. Two-call; the answer is `na * nr`.
///
/// # Safety
/// `raw` is readable for `na * nr` `sl_complex`; `params` points at one initialised
/// `sl_csa_params`; `out` is null or writable for `cap`; `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_chirp_scaling_focus(
    raw: *const SlComplex,
    na: usize,
    nr: usize,
    params: *const SlCsaParams,
    out: *mut SlComplex,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_chirp_scaling_focus";
        if raw.is_null() || params.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let n = match check_fft_aperture(WHO, na, nr) {
            Ok(n) => n,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states both are readable and initialised.
        let (aperture, p) = unsafe { (borrow_complex(raw, n), *params) };
        let focused = sig_csa_focus(
            &aperture,
            na,
            nr,
            &CsaParams {
                prf: p.prf,
                velocity: p.velocity,
                wavelength: p.wavelength,
                r_ref: p.r_ref,
                range_sample_rate: p.range_sample_rate,
                range_fm_rate: p.range_fm_rate,
            },
        );
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver_complex(&focused, out, cap, out_len) }
    })
}

/// Apply the chirp-scaling phase to one range line. Two-call.
///
/// # Safety
/// `line` is readable for `n` `sl_complex`; `out` is null or writable for `cap`; `out_len` is
/// writable for one `size_t`.
#[allow(clippy::too_many_arguments)] // the line, the chirp, the scaling and the sampling
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_apply_chirp_scaling(
    line: *const SlComplex,
    n: usize,
    chirp_rate: f64,
    scaling: f64,
    tau_ref: f64,
    sample_rate: f64,
    out: *mut SlComplex,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_apply_chirp_scaling";
        if line.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null line pointer"));
        }
        if let Err(status) = check_nonzero(WHO, "n", n)
            .and_then(|()| check_positive(WHO, "sample_rate", sample_rate, SL_ERR_DOMAIN))
        {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is readable for `n`.
        let samples = unsafe { borrow_complex(line, n) };
        let scaled = sig_apply_chirp_scaling(&samples, chirp_rate, scaling, tau_ref, sample_rate);
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver_complex(&scaled, out, cap, out_len) }
    })
}

/// The interferogram of two co-registered images: `s1 * conj(s2)`, sample by sample. Two-call.
///
/// # Safety
/// `s1` and `s2` are readable for `n` `sl_complex`; `out` is null or writable for `cap`; `out_len`
/// is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_interferogram(
    s1: *const SlComplex,
    s2: *const SlComplex,
    n: usize,
    out: *mut SlComplex,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_interferogram";
        if s1.is_null() || s2.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_nonzero(WHO, "n", n) {
            return status;
        }
        // SAFETY: checked non-null; the caller states both are readable for `n`.
        let (a, b) = unsafe { (borrow_complex(s1, n), borrow_complex(s2, n)) };
        let phase = sig_interferogram(&a, &b);
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver_complex(&phase, out, cap, out_len) }
    })
}

/// The windowed coherence of the same pair, one value per sample. Two-call.
///
/// # Safety
/// `s1` and `s2` are readable for `rows * cols` `sl_complex`; `out` is null or writable for `cap`
/// doubles; `out_len` is writable for one `size_t`.
#[allow(clippy::too_many_arguments)] // two images, their shape, the window and the two-call triple
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_coherence(
    s1: *const SlComplex,
    s2: *const SlComplex,
    rows: usize,
    cols: usize,
    half_window: usize,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_coherence";
        if s1.is_null() || s2.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let n = match check_raster(WHO, rows, cols) {
            Ok(n) => n,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states both are readable for `rows * cols`.
        let (a, b) = unsafe { (borrow_complex(s1, n), borrow_complex(s2, n)) };
        let gamma = sig_coherence(&a, &b, rows, cols, half_window);
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&gamma, out, cap, out_len) }
    })
}

/// Wrap a phase into `(-pi, pi]`. Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_wrap_phase(phi_rad: f64) -> f64 {
    guard_f64(|| sig_wrap_phase(phi_rad))
}

/// Unwrap a phase profile by accumulating branch jumps. Two-call.
///
/// # Safety
/// `wrapped` is readable for `n` doubles; `out` is null or writable for `cap`; `out_len` is
/// writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_unwrap_phase_1d(
    wrapped: *const f64,
    n: usize,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_unwrap_phase_1d";
        if wrapped.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: null wrapped pointer"),
            );
        }
        // SAFETY: checked non-null; the caller states it is readable for `n`.
        let phase = unsafe { core::slice::from_raw_parts(wrapped, n) };
        let unwrapped = sig_unwrap_1d(phase);
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&unwrapped, out, cap, out_len) }
    })
}

/// The three two-dimensional unwrappers, in the order they were published: quality-guided flood
/// fill, Goldstein branch cuts, and the weighted least-squares solution. Two-call, each.
///
/// The three disagree on purpose. Flood fill is fastest and follows the residues wherever they
/// lead; branch cuts refuse to cross a cut and so leave inconsistent regions alone; least squares
/// spreads the inconsistency over the whole raster instead of localising it.
///
/// # Safety
/// `wrapped` is readable for `rows * cols` doubles; `out` is null or writable for `cap`; `out_len`
/// is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_unwrap_phase_2d(
    wrapped: *const f64,
    rows: usize,
    cols: usize,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    // SAFETY: forwarding the caller's own contract on every pointer.
    unsafe {
        unwrap_raster(
            "sl_unwrap_phase_2d",
            wrapped,
            rows,
            cols,
            sig_unwrap_2d,
            out,
            cap,
            out_len,
        )
    }
}

/// See [`sl_unwrap_phase_2d`].
///
/// # Safety
/// As [`sl_unwrap_phase_2d`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_unwrap_phase_2d_branch_cut(
    wrapped: *const f64,
    rows: usize,
    cols: usize,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    // SAFETY: forwarding the caller's own contract on every pointer.
    unsafe {
        unwrap_raster(
            "sl_unwrap_phase_2d_branch_cut",
            wrapped,
            rows,
            cols,
            sig_unwrap_branch_cut,
            out,
            cap,
            out_len,
        )
    }
}

/// See [`sl_unwrap_phase_2d`].
///
/// # Safety
/// As [`sl_unwrap_phase_2d`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_unwrap_phase_2d_lstsq(
    wrapped: *const f64,
    rows: usize,
    cols: usize,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    // SAFETY: forwarding the caller's own contract on every pointer.
    unsafe {
        unwrap_raster(
            "sl_unwrap_phase_2d_lstsq",
            wrapped,
            rows,
            cols,
            sig_unwrap_lstsq,
            out,
            cap,
            out_len,
        )
    }
}

/// The three unwrappers differ only in which algorithm they call, so their checking is one function.
///
/// # Safety
/// `wrapped` is readable for `rows * cols` doubles; `out` is null or writable for `cap`; `out_len`
/// is writable for one `size_t`.
#[allow(clippy::too_many_arguments)] // the raster, its shape, the algorithm and the two-call triple
unsafe fn unwrap_raster(
    who: &str,
    wrapped: *const f64,
    rows: usize,
    cols: usize,
    algorithm: impl Fn(&[f64], usize, usize) -> Vec<f64>,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if wrapped.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{who}: null wrapped pointer"),
            );
        }
        let n = match check_raster(who, rows, cols) {
            Ok(n) => n,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states it is readable for `rows * cols`.
        let phase = unsafe { core::slice::from_raw_parts(wrapped, n) };
        let unwrapped = algorithm(phase, rows, cols);
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&unwrapped, out, cap, out_len) }
    })
}

/// The residue charge at each `2x2` loop of the wrapped raster: `+1`, `-1` or zero. Two-call.
///
/// A raster with no residues is one every unwrapper agrees on; the count of non-zero entries is
/// how hard the unwrapping is going to be. The charge is `int8_t` because it is one of three
/// values, and a wider type would invite a caller to believe otherwise.
///
/// # Safety
/// `wrapped` is readable for `rows * cols` doubles; `out` is null or writable for `cap` `int8_t`;
/// `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_residues(
    wrapped: *const f64,
    rows: usize,
    cols: usize,
    out: *mut i8,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_residues";
        if wrapped.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: null wrapped pointer"),
            );
        }
        let n = match check_raster(WHO, rows, cols) {
            Ok(n) => n,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states it is readable for `rows * cols`.
        let phase = unsafe { core::slice::from_raw_parts(wrapped, n) };
        let charges = sig_residues(phase, rows, cols);
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&charges, out, cap, out_len) }
    })
}

/// The Capon (minimum-variance) and Bartlett (beamforming) elevation spectra of one resolution
/// cell. Two-call; the answer is one value per requested elevation.
///
/// # Safety
/// `covariance` is readable for `baseline_count * baseline_count` `sl_complex`; `baselines` for
/// `baseline_count` doubles; `elevations` for `elevation_count`; `params` points at one
/// `sl_tomo_params`; `out` is null or writable for `cap`; `out_len` for one `size_t`.
#[allow(clippy::too_many_arguments)] // the covariance, two axes, the geometry and the two-call triple
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_capon_spectrum(
    covariance: *const SlComplex,
    baselines: *const f64,
    baseline_count: usize,
    elevations: *const f64,
    elevation_count: usize,
    params: *const SlTomoParams,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_capon_spectrum";
        let inputs = match unsafe {
            borrow_tomo(
                WHO,
                covariance,
                baselines,
                baseline_count,
                elevations,
                elevation_count,
                params,
            )
        } {
            Ok(inputs) => inputs,
            Err(status) => return status,
        };
        let spectrum = sig_capon(
            &inputs.covariance,
            &inputs.baselines,
            inputs.params.wavelength,
            inputs.params.slant_range,
            &inputs.elevations,
            inputs.params.loading,
        );
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&spectrum, out, cap, out_len) }
    })
}

/// See [`sl_capon_spectrum`]; this one ignores `loading`.
///
/// # Safety
/// As [`sl_capon_spectrum`].
#[allow(clippy::too_many_arguments)] // the covariance, two axes, the geometry and the two-call triple
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_bartlett_spectrum(
    covariance: *const SlComplex,
    baselines: *const f64,
    baseline_count: usize,
    elevations: *const f64,
    elevation_count: usize,
    params: *const SlTomoParams,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_bartlett_spectrum";
        let inputs = match unsafe {
            borrow_tomo(
                WHO,
                covariance,
                baselines,
                baseline_count,
                elevations,
                elevation_count,
                params,
            )
        } {
            Ok(inputs) => inputs,
            Err(status) => return status,
        };
        let spectrum = sig_bartlett(
            &inputs.covariance,
            &inputs.baselines,
            inputs.params.wavelength,
            inputs.params.slant_range,
            &inputs.elevations,
        );
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&spectrum, out, cap, out_len) }
    })
}

/// What a tomographic call borrows from its caller, once the shapes have been checked.
struct TomoInputs {
    covariance: Vec<SigComplex>,
    baselines: Vec<f64>,
    elevations: Vec<f64>,
    params: SlTomoParams,
}

/// A tomographic stack needs at least one baseline to sample and one elevation to sample at.
fn check_tomo_counts(who: &str, baseline_count: usize, elevation_count: usize) -> Result<(), i32> {
    check_nonzero(who, "baseline_count", baseline_count)?;
    check_nonzero(who, "elevation_count", elevation_count)
}

/// As [`check_tomo_counts`], for the three entry points that take a `baseline_count` square
/// covariance rather than a data vector, and answering how many entries that is.
fn check_tomo_covariance(
    who: &str,
    baseline_count: usize,
    elevation_count: usize,
) -> Result<usize, i32> {
    check_tomo_counts(who, baseline_count, elevation_count)?;
    check_raster(who, baseline_count, baseline_count)
}

/// The four tomographic entry points take the same arguments and make the same checks.
///
/// # Safety
/// As [`sl_capon_spectrum`], except that the first argument may be a data vector of
/// `baseline_count` rather than a covariance of its square, which is what SL1MMER takes.
#[allow(clippy::too_many_arguments)] // the same argument list the four entry points share
unsafe fn borrow_tomo(
    who: &str,
    covariance: *const SlComplex,
    baselines: *const f64,
    baseline_count: usize,
    elevations: *const f64,
    elevation_count: usize,
    params: *const SlTomoParams,
) -> Result<TomoInputs, i32> {
    if covariance.is_null() || baselines.is_null() || elevations.is_null() || params.is_null() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: null pointer"),
        ));
    }
    let entries = check_tomo_covariance(who, baseline_count, elevation_count)?;
    // SAFETY: checked non-null; the caller states each is readable for the length given.
    unsafe {
        Ok(TomoInputs {
            covariance: borrow_complex(covariance, entries),
            baselines: core::slice::from_raw_parts(baselines, baseline_count).to_vec(),
            elevations: core::slice::from_raw_parts(elevations, elevation_count).to_vec(),
            params: *params,
        })
    }
}

/// SL1MMER: the sparse elevation reconstruction, and the dense spectrum it is thresholded from.
/// Two-call, both.
///
/// Unlike the two spectra above, `y` is the *data* vector — one complex sample per baseline, not a
/// covariance — because SL1MMER solves for the scatterers directly rather than through a second
/// moment.
///
/// # Safety
/// `y` is readable for `baseline_count` `sl_complex`; `baselines` for `baseline_count` doubles;
/// `elevations` for `elevation_count`; `params` points at one `sl_tomo_params`; `out` is null or
/// writable for `cap`; `out_len` for one `size_t`.
#[allow(clippy::too_many_arguments)] // the data, two axes, the geometry and the two-call triple
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sl1mmer(
    y: *const SlComplex,
    baselines: *const f64,
    baseline_count: usize,
    elevations: *const f64,
    elevation_count: usize,
    params: *const SlTomoParams,
    out: *mut SlScatterer,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_sl1mmer";
        let inputs = match unsafe {
            borrow_sl1mmer(
                WHO,
                y,
                baselines,
                baseline_count,
                elevations,
                elevation_count,
                params,
            )
        } {
            Ok(inputs) => inputs,
            Err(status) => return status,
        };
        let scatterers: Vec<SlScatterer> = sig_sl1mmer(
            &inputs.covariance,
            &inputs.baselines,
            inputs.params.wavelength,
            inputs.params.slant_range,
            &inputs.elevations,
            &Sl1mmerConfig {
                lambda: inputs.params.lambda,
                max_iter: inputs.params.max_iter,
                threshold_rel: inputs.params.threshold_rel,
            },
        )
        .into_iter()
        .map(|s| SlScatterer {
            elevation_m: s.elevation,
            reflectivity: from_complex(s.reflectivity),
        })
        .collect();
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&scatterers, out, cap, out_len) }
    })
}

/// See [`sl_sl1mmer`]; this one answers the dense spectrum rather than the thresholded scatterers.
///
/// # Safety
/// As [`sl_sl1mmer`], with `out` writable for `cap` doubles.
#[allow(clippy::too_many_arguments)] // the data, two axes, the geometry and the two-call triple
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sl1mmer_spectrum(
    y: *const SlComplex,
    baselines: *const f64,
    baseline_count: usize,
    elevations: *const f64,
    elevation_count: usize,
    params: *const SlTomoParams,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_sl1mmer_spectrum";
        let inputs = match unsafe {
            borrow_sl1mmer(
                WHO,
                y,
                baselines,
                baseline_count,
                elevations,
                elevation_count,
                params,
            )
        } {
            Ok(inputs) => inputs,
            Err(status) => return status,
        };
        let spectrum = sig_sl1mmer_spectrum(
            &inputs.covariance,
            &inputs.baselines,
            inputs.params.wavelength,
            inputs.params.slant_range,
            &inputs.elevations,
            inputs.params.lambda,
            inputs.params.max_iter,
        );
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&spectrum, out, cap, out_len) }
    })
}

/// The SL1MMER pair's borrowing: the same as [`borrow_tomo`] except that the first array is one
/// sample per baseline rather than a full covariance.
///
/// # Safety
/// As [`sl_sl1mmer`].
#[allow(clippy::too_many_arguments)] // the same argument list the two entry points share
unsafe fn borrow_sl1mmer(
    who: &str,
    y: *const SlComplex,
    baselines: *const f64,
    baseline_count: usize,
    elevations: *const f64,
    elevation_count: usize,
    params: *const SlTomoParams,
) -> Result<TomoInputs, i32> {
    if y.is_null() || baselines.is_null() || elevations.is_null() || params.is_null() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: null pointer"),
        ));
    }
    check_tomo_counts(who, baseline_count, elevation_count)?;
    // SAFETY: checked non-null; the caller states each is readable for the length given.
    unsafe {
        Ok(TomoInputs {
            covariance: borrow_complex(y, baseline_count),
            baselines: core::slice::from_raw_parts(baselines, baseline_count).to_vec(),
            elevations: core::slice::from_raw_parts(elevations, elevation_count).to_vec(),
            params: *params,
        })
    }
}

/// The amplitude dispersion of every pixel in a stack — the PS-InSAR candidate score. The stack is
/// row-major `epoch * pixels + pixel`. Two-call; the answer is `pixels` long.
///
/// # Safety
/// `stack` is readable for `epochs * pixels` doubles; `out` is null or writable for `cap`;
/// `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_amplitude_dispersion(
    stack: *const f64,
    epochs: usize,
    pixels: usize,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_amplitude_dispersion";
        if stack.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: null stack pointer"),
            );
        }
        let n = match check_raster(WHO, epochs, pixels) {
            Ok(n) => n,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states it is readable for `epochs * pixels`.
        let flat = unsafe { core::slice::from_raw_parts(stack, n) };
        let rows: Vec<&[f64]> = flat.chunks_exact(pixels).collect();
        let dispersion = sig_amplitude_dispersion(&rows, pixels);
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&dispersion, out, cap, out_len) }
    })
}

/// The indices whose dispersion is below the threshold — the persistent scatterers. Two-call.
///
/// # Safety
/// `dispersion` is readable for `n` doubles; `out` is null or writable for `cap` `size_t`;
/// `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_select_persistent_scatterers(
    dispersion: *const f64,
    n: usize,
    threshold: f64,
    out: *mut usize,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_select_persistent_scatterers";
        if dispersion.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: null dispersion pointer"),
            );
        }
        // SAFETY: checked non-null; the caller states it is readable for `n`.
        let scores = unsafe { core::slice::from_raw_parts(dispersion, n) };
        let selected = sig_select_ps(scores, threshold);
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&selected, out, cap, out_len) }
    })
}

/// Invert an SBAS stack for the line-of-sight velocity and the residual topographic height.
///
/// # Safety
/// `phases`, `temporal_baselines` and `perp_baselines` are readable for `n` doubles; `out` is
/// writable for one `sl_deformation`.
#[allow(clippy::too_many_arguments)] // three stacks, their length, three geometry scalars and the answer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sbas_invert(
    phases: *const f64,
    temporal_baselines: *const f64,
    perp_baselines: *const f64,
    n: usize,
    wavelength: f64,
    slant_range: f64,
    incidence_rad: f64,
    out: *mut SlDeformation,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_sbas_invert";
        if phases.is_null()
            || temporal_baselines.is_null()
            || perp_baselines.is_null()
            || out.is_null()
        {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_nonzero(WHO, "n", n) {
            return status;
        }
        // SAFETY: checked non-null; the caller states each is readable for `n`.
        let (p, t, b) = unsafe {
            (
                core::slice::from_raw_parts(phases, n),
                core::slice::from_raw_parts(temporal_baselines, n),
                core::slice::from_raw_parts(perp_baselines, n),
            )
        };
        let (velocity_ms, height_m) =
            sig_sbas_invert(p, t, b, wavelength, slant_range, incidence_rad);
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe {
            *out = SlDeformation {
                velocity_ms,
                height_m,
            }
        };
        SL_OK
    })
}

/// The 3x3 coherency matrix of a set of Pauli scattering vectors, row-major. Two-call; the answer
/// is always `SL_COHERENCY_ENTRIES` long.
///
/// # Safety
/// `samples` is readable for `n` `sl_polarimetric_sample`; `out` is null or writable for `cap`
/// `sl_complex`; `out_len` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_coherency_matrix(
    samples: *const SlPolarimetricSample,
    n: usize,
    out: *mut SlComplex,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_coherency_matrix";
        if samples.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: null samples pointer"),
            );
        }
        if let Err(status) = check_nonzero(WHO, "n", n) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is readable for `n`.
        let looks: Vec<(SigComplex, SigComplex, SigComplex)> =
            unsafe { core::slice::from_raw_parts(samples, n) }
                .iter()
                .map(|s| (to_complex(s.s_hh), to_complex(s.s_hv), to_complex(s.s_vv)))
                .collect();
        let t3 = sig_coherency(&looks);
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver_complex(&t3, out, cap, out_len) }
    })
}

/// The Cloude–Pottier H/A/alpha decomposition of a 3x3 coherency matrix, row-major.
///
/// The name carries `_decompose` because C gives a typedef and a function one namespace, and the
/// struct it answers is already `sl_cloude_pottier`.
///
/// # Safety
/// `t3` is readable for `SL_COHERENCY_ENTRIES` `sl_complex`; `out` is writable for one
/// `sl_cloude_pottier`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_cloude_pottier_decompose(
    t3: *const SlComplex,
    out: *mut SlCloudePottier,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_cloude_pottier_decompose";
        if t3.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: checked non-null; the caller states it is readable for nine entries.
        let matrix = unsafe { borrow_complex(t3, COHERENCY_ENTRIES) };
        let (entropy, anisotropy, alpha_rad) = sig_cloude_pottier(&matrix);
        // SAFETY: checked non-null; written only now that the answer exists.
        unsafe {
            *out = SlCloudePottier {
                entropy,
                anisotropy,
                alpha_rad,
            }
        };
        SL_OK
    })
}

/// The three Pauli channel powers — surface, double-bounce and volume — of one scattering vector.
///
/// # Safety
/// `sample` points at one initialised `sl_polarimetric_sample`; `out` is writable for three doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_pauli_powers(
    sample: *const SlPolarimetricSample,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_pauli_powers";
        if sample.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let s = unsafe { *sample };
        let p = sig_pauli_powers(to_complex(s.s_hh), to_complex(s.s_hv), to_complex(s.s_vv));
        // SAFETY: checked non-null; the caller states it is writable for three doubles.
        unsafe { core::ptr::copy_nonoverlapping(p.as_ptr(), out, 3) };
        SL_OK
    })
}

/// The 3 dB main-lobe width of an impulse response, in samples, and its peak-to-sidelobe ratio in
/// decibels — the two numbers a compression is judged by.
///
/// A profile with no resolvable main lobe has neither, which is `SL_ERR_DOMAIN`. The Python bridge
/// used to panic here; a C caller has no exception to catch, so a status is the only honest answer.
///
/// # Safety
/// `magnitude` is readable for `n` doubles; `out` is writable for one double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_resolution_3db_samples(
    magnitude: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver_response("sl_resolution_3db_samples", magnitude, n, sig_res3db, out) }
    })
}

/// See [`sl_resolution_3db_samples`].
///
/// # Safety
/// As [`sl_resolution_3db_samples`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_peak_sidelobe_ratio_db(
    magnitude: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver_response("sl_peak_sidelobe_ratio_db", magnitude, n, sig_pslr, out) }
    })
}

/// The two impulse-response metrics differ only in which one they call.
///
/// # Safety
/// `magnitude` is readable for `n` doubles; `out` is writable for one double.
unsafe fn deliver_response(
    who: &str,
    magnitude: *const f64,
    n: usize,
    metric: impl Fn(&[f64]) -> f64,
    out: *mut f64,
) -> i32 {
    if magnitude.is_null() || out.is_null() {
        return fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: null pointer"));
    }
    if let Err(status) = check_nonzero(who, "n", n) {
        return status;
    }
    // SAFETY: checked non-null; the caller states it is readable for `n`.
    let profile = unsafe { core::slice::from_raw_parts(magnitude, n) };
    if let Err(status) = check_all_finite(who, "magnitude", profile) {
        return status;
    }
    let value = metric(profile);
    if !value.is_finite() {
        return fail(
            SL_ERR_DOMAIN,
            format!("{who}: this profile has no resolvable main lobe"),
        );
    }
    // SAFETY: checked non-null.
    unsafe { *out = value };
    SL_OK
}

// ---------------------------------------------------------------------------------------------------
// Phase-8 earthvision and deepvision -- multi-view geometry, stereo, BRDF, pan-sharpening, detection.
// ---------------------------------------------------------------------------------------------------

/// `sl_camera`: a calibrated pinhole camera, as the collinearity equations take it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlCamera {
    /// Rotation world to camera, row-major.
    pub rotation: [f64; 9],
    /// Camera centre in world coordinates.
    pub center: [f64; 3],
    /// Focal length, pixels.
    pub focal: f64,
    /// Principal point `[c_x, c_y]`, pixels.
    pub principal: [f64; 2],
}

/// `sl_mvs_view`: one image and the camera that took it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlMvsView {
    /// The image, row-major, `rows * cols` doubles. Borrowed for the call and not retained.
    pub image: *const f64,
    /// The camera that took it.
    pub camera: SlCamera,
}

/// `sl_sgm_config`: the tuning of a Semi-Global Match.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlSgmConfig {
    /// Disparities searched, `d` in `[0, d_max)`.
    pub d_max: usize,
    /// Census window radius; the window side is `2 * radius + 1`.
    pub radius: usize,
    /// Penalty for a one-step disparity change -- what lets a slanted surface stay smooth.
    pub p1: f64,
    /// Penalty for a larger jump -- what lets a depth discontinuity survive.
    pub p2: f64,
}

/// `sl_harris_config`: the five knobs of the corner detector.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlHarrisConfig {
    /// The `k` of the Harris response `det - k * trace^2`.
    pub k: f64,
    /// Structure-tensor window radius.
    pub window_radius: usize,
    /// Keep a corner whose response is at least this fraction of the strongest.
    pub threshold_rel: f64,
    /// Non-maximum suppression radius.
    pub nms_radius: usize,
    /// Cap on how many corners are returned.
    pub max_corners: usize,
}

/// `sl_corner`: one detected corner and how strong it was.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlCorner {
    /// Pixel column.
    pub x: f64,
    /// Pixel row.
    pub y: f64,
    /// Harris response.
    pub response: f64,
}

/// `sl_feature_match`: the same feature seen in two views.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlFeatureMatch {
    /// Index of the first view.
    pub view_a: usize,
    /// Index of the corner within that view.
    pub corner_a: usize,
    /// Index of the second view.
    pub view_b: usize,
    /// Index of the corner within that view.
    pub corner_b: usize,
}

/// `sl_track_point`: what a feature track triangulated to, or that it did not.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlTrackPoint {
    /// The world point, meaningful only when `resolved`.
    pub position: SlVec3,
    /// False when the track's views are degenerate -- parallel rays have no intersection.
    pub resolved: bool,
}

/// `sl_ground_control_point`: a ground point whose image coordinates are known.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlGroundControlPoint {
    /// Degrees.
    pub latitude: f64,
    /// Degrees.
    pub longitude: f64,
    /// Metres above the ellipsoid.
    pub height: f64,
    /// Measured image row, pixels.
    pub row: f64,
    /// Measured image column, pixels.
    pub col: f64,
}

/// `sl_image_point`: a pixel, as a sensor model answers one.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlImagePoint {
    /// Image row, pixels.
    pub row: f64,
    /// Image column, pixels.
    pub col: f64,
}

/// `sl_atmospheric_coupling`: the four 6S terms that stand between a surface and a sensor.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlAtmosphericCoupling {
    /// What the atmosphere itself reflects, before the ground is reached.
    pub path_reflectance: f64,
    /// Downward transmittance.
    pub transmittance_down: f64,
    /// Upward transmittance.
    pub transmittance_up: f64,
    /// Spherical albedo of the atmosphere, seen from below.
    pub spherical_albedo: f64,
}

/// `sl_bbox`: an axis-aligned detection box.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlBbox {
    /// Left edge.
    pub x1: f64,
    /// Top edge.
    pub y1: f64,
    /// Right edge.
    pub x2: f64,
    /// Bottom edge.
    pub y2: f64,
}

/// The crate's own camera, from the C one.
fn camera_of(c: SlCamera) -> EvCamera {
    EvCamera {
        rotation: Mat3::from_rows([
            [c.rotation[0], c.rotation[1], c.rotation[2]],
            [c.rotation[3], c.rotation[4], c.rotation[5]],
            [c.rotation[6], c.rotation[7], c.rotation[8]],
        ]),
        center: Vec3::new(c.center[0], c.center[1], c.center[2]),
        focal: c.focal,
        principal: c.principal,
    }
}

/// The raster shape a caller claims, checked against the buffer it was handed.
fn check_raster(who: &str, rows: usize, cols: usize) -> Result<usize, i32> {
    match rows.checked_mul(cols) {
        Some(0) | None => Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: the raster is {rows} x {cols}, which is not an image"),
        )),
        Some(n) => Ok(n),
    }
}

/// The SGM tuning a rectified pair is matched with.
///
/// The census window is packed one bit per neighbour into a u64, so a radius past 3 has more
/// neighbours than there are bits. The crate shifts them off the top silently; a C caller who
/// asked for a 9x9 window deserves to be told it does not exist.
fn check_sgm_config(who: &str, cfg: &SlSgmConfig) -> Result<(), i32> {
    check_nonzero(who, "d_max", cfg.d_max)?;
    if cfg.radius > 3 {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!(
                "{who}: a census radius of {} needs more than the 64 bits the descriptor has",
                cfg.radius
            ),
        ));
    }
    Ok(())
}

/// Dense disparity by Semi-Global Matching of a rectified pair. Two-call; the answer is
/// `rows * cols` disparities, row-major, one per pixel of the left image.
///
/// # Safety
/// `left` and `right` are readable for `rows * cols` doubles; `config` points at one initialised
/// `sl_sgm_config`; `out` is null or writable for `cap`; `out_len` is writable for one `size_t`.
#[allow(clippy::too_many_arguments)] // the pair, its shape, the tuning, and the two-call triple
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_semi_global_match(
    left: *const f64,
    right: *const f64,
    rows: usize,
    cols: usize,
    config: *const SlSgmConfig,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_semi_global_match";
        if left.is_null() || right.is_null() || config.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let n = match check_raster(WHO, rows, cols) {
            Ok(n) => n,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states it is initialised.
        let cfg = unsafe { *config };
        if let Err(status) = check_sgm_config(WHO, &cfg) {
            return status;
        }
        // SAFETY: checked non-null; the caller states both are readable for `rows * cols`.
        let (l, r) = unsafe {
            (
                core::slice::from_raw_parts(left, n),
                core::slice::from_raw_parts(right, n),
            )
        };
        let disparity = ev_sgm(
            l,
            r,
            rows,
            cols,
            &SgmConfig {
                d_max: cfg.d_max,
                radius: cfg.radius,
                p1: cfg.p1,
                p2: cfg.p2,
            },
        );
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&disparity, out, cap, out_len) }
    })
}

/// Dense depth by plane-sweep multi-view stereo. Two-call; the answer is `rows * cols` depths in
/// the reference camera's frame, with zero where no hypothesis was photo-consistent.
///
/// # Safety
/// `reference` and each of the `neighbor_count` `neighbors` hold an `image` readable for
/// `rows * cols` doubles; `depths` is readable for `depth_count`; `out` is null or writable for
/// `cap`; `out_len` is writable for one `size_t`.
#[allow(clippy::too_many_arguments)] // the views, the shape, the sweep, and the two-call triple
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_plane_sweep_mvs(
    reference: *const SlMvsView,
    neighbors: *const SlMvsView,
    neighbor_count: usize,
    rows: usize,
    cols: usize,
    depths: *const f64,
    depth_count: usize,
    patch_radius: usize,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_plane_sweep_mvs";
        if reference.is_null() || neighbors.is_null() || depths.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let n = match check_raster(WHO, rows, cols) {
            Ok(n) => n,
            Err(status) => return status,
        };
        if let Err(status) = check_nonzero(WHO, "neighbor_count", neighbor_count)
            .and_then(|()| check_nonzero(WHO, "depth_count", depth_count))
        {
            return status;
        }
        // SAFETY: checked non-null; the caller states they are readable and initialised.
        let (base, views, sweep) = unsafe {
            (
                *reference,
                core::slice::from_raw_parts(neighbors, neighbor_count),
                core::slice::from_raw_parts(depths, depth_count),
            )
        };
        if base.image.is_null() || views.iter().any(|v| v.image.is_null()) {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: a view carries a null image"),
            );
        }
        // SAFETY: each view states its image is readable for `rows * cols`.
        let reference_image = unsafe { core::slice::from_raw_parts(base.image, n) };
        let cameras: Vec<EvCamera> = views.iter().map(|v| camera_of(v.camera)).collect();
        // SAFETY: as above, one image per view.
        let images: Vec<&[f64]> = views
            .iter()
            .map(|v| unsafe { core::slice::from_raw_parts(v.image, n) })
            .collect();
        let pairs: Vec<(&[f64], &EvCamera)> = images.iter().copied().zip(cameras.iter()).collect();
        let depth_map = ev_plane_sweep(
            reference_image,
            &camera_of(base.camera),
            &pairs,
            rows,
            cols,
            sweep,
            patch_radius,
        );
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&depth_map, out, cap, out_len) }
    })
}

/// Triangulate one world point from two or more views of it.
///
/// Rays that do not meet are a degenerate configuration rather than a bad input, so a pair whose
/// baselines are parallel is `SL_ERR_DOMAIN`: there is no point to answer, which is a different
/// statement from having computed one badly.
///
/// # Safety
/// `cameras` is readable for `n` `sl_camera`; `pixels` for `2 * n` doubles, `(u, v)` per camera;
/// `out` is writable for one `sl_vec3`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_triangulate(
    cameras: *const SlCamera,
    pixels: *const f64,
    n: usize,
    out: *mut SlVec3,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_triangulate";
        if cameras.is_null() || pixels.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_at_least(WHO, "n", n, 2) {
            return status;
        }
        // SAFETY: checked non-null; the caller states both are readable.
        let (cams, uv) = unsafe {
            (
                core::slice::from_raw_parts(cameras, n),
                core::slice::from_raw_parts(pixels, 2 * n),
            )
        };
        let owned: Vec<EvCamera> = cams.iter().map(|&c| camera_of(c)).collect();
        let views: Vec<(&EvCamera, [f64; 2])> = owned
            .iter()
            .enumerate()
            .map(|(i, c)| (c, [uv[2 * i], uv[2 * i + 1]]))
            .collect();
        let Some(point) = ev_triangulate(&views) else {
            return fail(
                SL_ERR_DOMAIN,
                format!("{WHO}: the {n} views are degenerate and their rays do not intersect"),
            );
        };
        // SAFETY: checked non-null.
        unsafe {
            *out = SlVec3 {
                x: point.x,
                y: point.y,
                z: point.z,
            };
        }
        SL_OK
    })
}

/// Build feature tracks across views and triangulate each one. Two-call; the answer is one
/// `sl_track_point` per track, in the order the tracks were built.
///
/// Every match must name a view that exists and, within it, a corner that exists.
///
/// The track builder indexes both directly and would panic otherwise, and a panic is not a value a
/// C caller can receive.
fn check_matches(who: &str, pairs: &[SlFeatureMatch], counts: &[usize]) -> Result<(), i32> {
    let corners = |view: usize| counts.get(view).copied().unwrap_or(0);
    if let Some(bad) = pairs
        .iter()
        .find(|m| m.corner_a >= corners(m.view_a) || m.corner_b >= corners(m.view_b))
    {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!(
                "{who}: a match names view {} corner {} against view {} corner {}, and one of \
                 those does not exist",
                bad.view_a, bad.corner_a, bad.view_b, bad.corner_b
            ),
        ));
    }
    Ok(())
}

/// How long the concatenated corner list is, refusing counts that sum past what can be indexed.
fn check_corner_total(who: &str, counts: &[usize]) -> Result<usize, i32> {
    counts
        .iter()
        .try_fold(0_usize, |a, &b| a.checked_add(b))
        .ok_or_else(|| {
            fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{who}: the corner counts sum past a size_t"),
            )
        })
}

/// `corners` is the views' corner lists laid end to end and `corners_per_view` says how long each
/// one is, because C has no ragged array. A match that names a view or a corner outside those
/// lists is refused here: the track builder indexes them directly and would panic, and a panic is
/// not a value a C caller can receive.
///
/// # Safety
/// `corners` is readable for the sum of `corners_per_view`, which is readable for `view_count`;
/// `matches` for `match_count`; `cameras` for `view_count`; `out` is null or writable for `cap`;
/// `out_len` is writable for one `size_t`.
#[allow(clippy::too_many_arguments)] // the ragged corner lists, the matches, and the two-call triple
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_sfm_triangulate(
    corners: *const SlCorner,
    corners_per_view: *const usize,
    view_count: usize,
    matches: *const SlFeatureMatch,
    match_count: usize,
    cameras: *const SlCamera,
    out: *mut SlTrackPoint,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_sfm_triangulate";
        if corners_per_view.is_null() || cameras.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_nonzero(WHO, "view_count", view_count) {
            return status;
        }
        // SAFETY: checked non-null; the caller states they are readable for `view_count`.
        let (counts, cams) = unsafe {
            (
                core::slice::from_raw_parts(corners_per_view, view_count),
                core::slice::from_raw_parts(cameras, view_count),
            )
        };
        let total = match check_corner_total(WHO, counts) {
            Ok(total) => total,
            Err(status) => return status,
        };
        if total != 0 && corners.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: {total} corners were promised and none were given"),
            );
        }
        if match_count != 0 && matches.is_null() {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: {match_count} matches were promised and none were given"),
            );
        }
        // SAFETY: checked non-null against the lengths the caller stated.
        let (flat, pairs) = unsafe {
            (
                if total == 0 {
                    &[][..]
                } else {
                    core::slice::from_raw_parts(corners, total)
                },
                if match_count == 0 {
                    &[][..]
                } else {
                    core::slice::from_raw_parts(matches, match_count)
                },
            )
        };
        let mut lists: Vec<Vec<EvCorner>> = Vec::with_capacity(view_count);
        let mut at = 0;
        for &count in counts {
            lists.push(
                flat[at..at + count]
                    .iter()
                    .map(|c| EvCorner {
                        x: c.x,
                        y: c.y,
                        response: c.response,
                    })
                    .collect(),
            );
            at += count;
        }
        if let Err(status) = check_matches(WHO, pairs, counts) {
            return status;
        }
        let borrowed: Vec<&[EvCorner]> = lists.iter().map(Vec::as_slice).collect();
        let tuples: Vec<(usize, usize, usize, usize)> = pairs
            .iter()
            .map(|m| (m.view_a, m.corner_a, m.view_b, m.corner_b))
            .collect();
        let tracks: Vec<Track> = build_tracks(&borrowed, &tuples);
        let owned: Vec<EvCamera> = cams.iter().map(|&c| camera_of(c)).collect();
        let points: Vec<SlTrackPoint> = tracks
            .iter()
            .map(|t| {
                let views: Vec<(&EvCamera, [f64; 2])> = t
                    .observations
                    .iter()
                    .map(|o: &Observation2d| (&owned[o.view], [o.x, o.y]))
                    .collect();
                match ev_triangulate(&views) {
                    Some(p) => SlTrackPoint {
                        position: SlVec3 {
                            x: p.x,
                            y: p.y,
                            z: p.z,
                        },
                        resolved: true,
                    },
                    None => SlTrackPoint {
                        position: SlVec3 {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                        },
                        resolved: false,
                    },
                }
            })
            .collect();
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&points, out, cap, out_len) }
    })
}

/// Harris corners of one grayscale image. Two-call; the answer is however many cleared the
/// threshold, at most `max_corners`.
///
/// # Safety
/// `image` is readable for `rows * cols` doubles; `config` points at one initialised
/// `sl_harris_config`; `out` is null or writable for `cap`; `out_len` is writable for one `size_t`.
#[allow(clippy::too_many_arguments)] // the image, its shape, the tuning, and the two-call triple
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_harris_corners(
    image: *const f64,
    rows: usize,
    cols: usize,
    config: *const SlHarrisConfig,
    out: *mut SlCorner,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_harris_corners";
        if image.is_null() || config.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let n = match check_raster(WHO, rows, cols) {
            Ok(n) => n,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states both are readable and initialised.
        let (data, cfg) = unsafe { (core::slice::from_raw_parts(image, n), *config) };
        let corners = detect_corners(
            EvImage {
                data,
                h: rows,
                w: cols,
            },
            &HarrisConfig {
                k: cfg.k,
                window_radius: cfg.window_radius,
                threshold_rel: cfg.threshold_rel,
                nms_radius: cfg.nms_radius,
                max_corners: cfg.max_corners,
            },
        );
        let answer: Vec<SlCorner> = corners
            .iter()
            .map(|c| SlCorner {
                x: c.x,
                y: c.y,
                response: c.response,
            })
            .collect();
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&answer, out, cap, out_len) }
    })
}

/// Fit a Grodecki-Dial bias correction to an RPC from ground control, and answer where the
/// corrected model puts one ground point.
///
/// `affine` chooses the six-parameter fit over the two shifts; the affine one needs three control
/// points to be determined and the shift one needs one. `rpb_text` is the `.RPB` the vendor ships,
/// passed as text with its length rather than as a path, because this library does not read files.
///
/// # Safety
/// `rpb_text` is readable for `text_len` bytes; `gcps` for `n` `sl_ground_control_point`; `out` is
/// writable for one `sl_image_point`.
#[allow(clippy::too_many_arguments)] // the model, the control, the query point, and the fit choice
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_rpc_block_adjust(
    rpb_text: *const c_char,
    text_len: usize,
    gcps: *const SlGroundControlPoint,
    n: usize,
    latitude: f64,
    longitude: f64,
    height: f64,
    affine: bool,
    out: *mut SlImagePoint,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_rpc_block_adjust";
        if gcps.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: forwarding the caller's own contract.
        let Some(text) = (unsafe { borrow_sized_str(rpb_text, text_len) }) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: the RPB text is null or is not UTF-8"),
            );
        };
        let Some(rpc) = RationalPolynomialCamera::parse_rpb(text) else {
            return fail(
                SL_ERR_PARSE,
                format!("{WHO}: the text is not an RPB the parser recognises"),
            );
        };
        let model = if affine {
            AdjustmentModel::Affine
        } else {
            AdjustmentModel::Shift
        };
        let needed = if affine { 3 } else { 1 };
        if let Err(status) = check_at_least(WHO, "n", n, needed) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is readable for `n`.
        let control: Vec<GroundControlPoint> = unsafe { core::slice::from_raw_parts(gcps, n) }
            .iter()
            .map(|g| GroundControlPoint {
                latitude: g.latitude,
                longitude: g.longitude,
                height: g.height,
                row: g.row,
                col: g.col,
            })
            .collect();
        let adjustment = estimate_adjustment(&rpc, &control, model);
        let (row, col) = forward_adjusted(&rpc, latitude, longitude, height, &adjustment);
        // SAFETY: checked non-null.
        unsafe { *out = SlImagePoint { row, col } };
        SL_OK
    })
}

/// The two MODIS RTLSR kernels and the reflectance they combine into. All three are total: the
/// kernels are trigonometry and have an answer at every angle.
#[unsafe(no_mangle)]
pub extern "C" fn sl_ross_thick_kernel(theta_s: f64, theta_v: f64, rel_azimuth: f64) -> f64 {
    guard_f64(|| ev_ross_thick(theta_s, theta_v, rel_azimuth))
}

/// See [`sl_ross_thick_kernel`].
#[unsafe(no_mangle)]
pub extern "C" fn sl_li_sparse_reciprocal_kernel(
    theta_s: f64,
    theta_v: f64,
    rel_azimuth: f64,
) -> f64 {
    guard_f64(|| ev_li_sparse(theta_s, theta_v, rel_azimuth))
}

/// See [`sl_ross_thick_kernel`]; this one is `f_iso + f_vol * K_vol + f_geo * K_geo`.
#[unsafe(no_mangle)]
pub extern "C" fn sl_brdf_reflectance(
    f_iso: f64,
    f_vol: f64,
    f_geo: f64,
    theta_s: f64,
    theta_v: f64,
    rel_azimuth: f64,
) -> f64 {
    guard_f64(|| ev_brdf(f_iso, f_vol, f_geo, theta_s, theta_v, rel_azimuth))
}

/// The 6S forward and inverse: what a surface of a given reflectance shows at the top of the
/// atmosphere, and what a top-of-atmosphere reflectance implies about the surface. Total, and
/// exact inverses of one another. A null coupling answers NaN.
///
/// # Safety
/// `coupling` is null, or points at one initialised `sl_atmospheric_coupling`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_atmospheric_toa_from_surface(
    coupling: *const SlAtmosphericCoupling,
    surface_reflectance: f64,
) -> f64 {
    guard_f64(|| {
        // SAFETY: forwarding the caller's own contract.
        match unsafe { coupling_of("sl_atmospheric_toa_from_surface", coupling) } {
            Some(c) => c.toa_from_surface(surface_reflectance),
            None => f64::NAN,
        }
    })
}

/// See [`sl_atmospheric_toa_from_surface`].
///
/// # Safety
/// As [`sl_atmospheric_toa_from_surface`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_atmospheric_surface_from_toa(
    coupling: *const SlAtmosphericCoupling,
    toa_reflectance: f64,
) -> f64 {
    guard_f64(|| {
        // SAFETY: forwarding the caller's own contract.
        match unsafe { coupling_of("sl_atmospheric_surface_from_toa", coupling) } {
            Some(c) => c.surface_from_toa(toa_reflectance),
            None => f64::NAN,
        }
    })
}

/// The crate's own atmospheric coupling, from the C one, or nothing if the pointer was null.
///
/// # Safety
/// `p` is null, or points at one initialised `sl_atmospheric_coupling`.
unsafe fn coupling_of(who: &str, p: *const SlAtmosphericCoupling) -> Option<AtmosphericCoupling> {
    if p.is_null() {
        let _ = fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: null coupling"));
        return None;
    }
    // SAFETY: checked non-null; the caller states it is initialised.
    let c = unsafe { *p };
    Some(AtmosphericCoupling {
        path_reflectance: c.path_reflectance,
        transmittance_down: c.transmittance_down,
        transmittance_up: c.transmittance_up,
        spherical_albedo: c.spherical_albedo,
    })
}

/// The two pan-sharpeners: Brovey's ratio substitution and Gram-Schmidt component substitution.
/// `ms_bands` is `band_count` bands of `pixels` samples laid end to end, and the answer has the
/// same shape. Two-call, both.
///
/// # Safety
/// `ms_bands` is readable for `band_count * pixels` doubles and `pan` for `pixels`; `out` is null
/// or writable for `cap`; `out_len` is writable for one `size_t`.
#[allow(clippy::too_many_arguments)] // the bands, their shape, the pan, and the two-call triple
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_brovey_pan_sharpen(
    ms_bands: *const f64,
    band_count: usize,
    pixels: usize,
    pan: *const f64,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: forwarding the caller's own contract.
        unsafe {
            deliver_pan_sharpened(
                "sl_brovey_pan_sharpen",
                ms_bands,
                band_count,
                pixels,
                pan,
                ev_brovey,
                out,
                cap,
                out_len,
            )
        }
    })
}

/// See [`sl_brovey_pan_sharpen`].
///
/// # Safety
/// As [`sl_brovey_pan_sharpen`].
#[allow(clippy::too_many_arguments)] // as `sl_brovey_pan_sharpen`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_gram_schmidt_pan_sharpen(
    ms_bands: *const f64,
    band_count: usize,
    pixels: usize,
    pan: *const f64,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: forwarding the caller's own contract.
        unsafe {
            deliver_pan_sharpened(
                "sl_gram_schmidt_pan_sharpen",
                ms_bands,
                band_count,
                pixels,
                pan,
                ev_gram_schmidt,
                out,
                cap,
                out_len,
            )
        }
    })
}

/// The two pan-sharpeners differ only in which one they call.
///
/// # Safety
/// As [`sl_brovey_pan_sharpen`].
#[allow(clippy::too_many_arguments)] // as the two entry points it serves
unsafe fn deliver_pan_sharpened(
    who: &str,
    ms_bands: *const f64,
    band_count: usize,
    pixels: usize,
    pan: *const f64,
    sharpen: impl Fn(&[&[f64]], &[f64]) -> Vec<Vec<f64>>,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    if ms_bands.is_null() || pan.is_null() {
        return fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: null pointer"));
    }
    let total = match check_raster(who, band_count, pixels) {
        Ok(n) => n,
        Err(status) => return status,
    };
    // SAFETY: checked non-null; the caller states the two lengths.
    let (flat, panchromatic) = unsafe {
        (
            core::slice::from_raw_parts(ms_bands, total),
            core::slice::from_raw_parts(pan, pixels),
        )
    };
    let bands: Vec<&[f64]> = (0..band_count)
        .map(|k| &flat[k * pixels..(k + 1) * pixels])
        .collect();
    let sharpened: Vec<f64> = sharpen(&bands, panchromatic).concat();
    // SAFETY: forwarding the caller's own contract.
    unsafe { deliver(&sharpened, out, cap, out_len) }
}

/// Greedy non-maximum suppression of detection boxes. Two-call; the answer is the kept indices in
/// descending score order.
///
/// # Safety
/// `boxes` is readable for `n` `sl_bbox` and `scores` for `n` doubles; `out` is null or writable
/// for `cap` `size_t`; `out_len` is writable for one `size_t`.
#[allow(clippy::too_many_arguments)] // the boxes, their scores, the threshold, and the triple
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_nms(
    boxes: *const SlBbox,
    scores: *const f64,
    n: usize,
    iou_threshold: f64,
    out: *mut usize,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_nms";
        if n != 0 && (boxes.is_null() || scores.is_null()) {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: checked non-null against the length the caller stated.
        let (bs, ss) = unsafe {
            if n == 0 {
                (&[][..], &[][..])
            } else {
                (
                    core::slice::from_raw_parts(boxes, n),
                    core::slice::from_raw_parts(scores, n),
                )
            }
        };
        let owned: Vec<BBox> = bs
            .iter()
            .map(|b| BBox::new(b.x1, b.y1, b.x2, b.y2))
            .collect();
        let kept = dv_nms(&owned, ss, iou_threshold);
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&kept, out, cap, out_len) }
    })
}

/// The intersection-over-union of two boxes. Total; a null answers NaN.
///
/// # Safety
/// `a` and `b` are null, or each points at one initialised `sl_bbox`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_box_iou(a: *const SlBbox, b: *const SlBbox) -> f64 {
    guard_f64(|| {
        if a.is_null() || b.is_null() {
            let _ = fail(SL_ERR_INVALID_ARGUMENT, "sl_box_iou: null box pointer");
            return f64::NAN;
        }
        // SAFETY: checked non-null; the caller states both are initialised.
        let (first, second) = unsafe { (*a, *b) };
        BBox::new(first.x1, first.y1, first.x2, first.y2)
            .iou(&BBox::new(second.x1, second.y1, second.x2, second.y2))
    })
}

/// Decode a predicted delta against its anchor, in the Faster R-CNN centre parameterisation.
///
/// # Safety
/// `anchor` points at one initialised `sl_bbox`; `delta` is readable for four doubles
/// `(dx, dy, dw, dh)`; `out` is writable for one `sl_bbox`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_decode_box(
    anchor: *const SlBbox,
    delta: *const f64,
    out: *mut SlBbox,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_decode_box";
        if anchor.is_null() || delta.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: checked non-null; the caller states both are readable.
        let (a, d) = unsafe { (*anchor, core::slice::from_raw_parts(delta, 4)) };
        let decoded = dv_decode_box(&BBox::new(a.x1, a.y1, a.x2, a.y2), [d[0], d[1], d[2], d[3]]);
        // SAFETY: checked non-null.
        unsafe {
            *out = SlBbox {
                x1: decoded.x1,
                y1: decoded.y1,
                x2: decoded.x2,
                y2: decoded.y2,
            };
        }
        SL_OK
    })
}

/// Lay anchor boxes over a feature map. Two-call; the answer is
/// `feat_rows * feat_cols * scale_count * ratio_count` boxes.
///
/// # Safety
/// `scales` is readable for `scale_count` doubles and `aspect_ratios` for `ratio_count`; `out` is
/// null or writable for `cap`; `out_len` is writable for one `size_t`.
#[allow(clippy::too_many_arguments)] // the map, the stride, the two grids, and the two-call triple
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_generate_anchors(
    feat_rows: usize,
    feat_cols: usize,
    stride: f64,
    scales: *const f64,
    scale_count: usize,
    aspect_ratios: *const f64,
    ratio_count: usize,
    out: *mut SlBbox,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_generate_anchors";
        if scales.is_null() || aspect_ratios.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_nonzero(WHO, "scale_count", scale_count)
            .and_then(|()| check_nonzero(WHO, "ratio_count", ratio_count))
        {
            return status;
        }
        // SAFETY: checked non-null; the caller states both are readable.
        let (s, r) = unsafe {
            (
                core::slice::from_raw_parts(scales, scale_count),
                core::slice::from_raw_parts(aspect_ratios, ratio_count),
            )
        };
        let anchors: Vec<SlBbox> = dv_generate_anchors(feat_rows, feat_cols, stride, s, r)
            .into_iter()
            .map(|b| SlBbox {
                x1: b.x1,
                y1: b.y1,
                x2: b.x2,
                y2: b.y2,
            })
            .collect();
        // SAFETY: forwarding the caller's own contract.
        unsafe { deliver(&anchors, out, cap, out_len) }
    })
}

/// `sl_gpt3_weather`: everything the GPT3 climatology knows about one point and one day.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlGpt3Weather {
    /// Hydrostatic VMF3 mapping coefficient `a_h`.
    pub ah: f64,
    /// Wet VMF3 mapping coefficient `a_w`.
    pub aw: f64,
    /// Total surface pressure, Pa.
    pub pressure: f64,
    /// Surface temperature, K.
    pub temperature: f64,
    /// Water-vapour partial pressure, Pa.
    pub water_vapor_pressure: f64,
    /// Weighted mean temperature, K.
    pub tm: f64,
    /// Water-vapour decrease factor.
    pub lambda: f64,
    /// Hydrostatic north gradient coefficient.
    pub gnh: f64,
    /// Hydrostatic east gradient coefficient.
    pub geh: f64,
    /// Wet north gradient coefficient.
    pub gnw: f64,
    /// Wet east gradient coefficient.
    pub gew: f64,
}

/// `sl_vmf_coefficients`: the Vienna mapping coefficients and the zenith delays shipped with them.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlVmfCoefficients {
    /// Hydrostatic mapping coefficient `a_h`.
    pub ah: f64,
    /// Wet mapping coefficient `a_w`.
    pub aw: f64,
    /// Zenith hydrostatic delay, metres.
    pub zhd: f64,
    /// Zenith wet delay, metres.
    pub zwd: f64,
}

/// The GPT3 empirical troposphere at one point and one day of the year.
///
/// Blind in the sense that matters: it reads no file and needs no forecast, only the 5x5 degree
/// climatology compiled into the library.
///
/// # Safety
/// `out` is writable for one `sl_gpt3_weather`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_gpt3(
    latitude_rad: f64,
    longitude_rad: f64,
    day_of_year: f64,
    height_m: f64,
    out: *mut SlGpt3Weather,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_gpt3";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out pointer"));
        }
        let r = gnss_gpt3(latitude_rad, longitude_rad, day_of_year, height_m);
        // SAFETY: checked non-null.
        unsafe {
            *out = SlGpt3Weather {
                ah: r.ah,
                aw: r.aw,
                pressure: r.pressure,
                temperature: r.temperature,
                water_vapor_pressure: r.water_vapor_pressure,
                tm: r.tm,
                lambda: r.lambda,
                gnh: r.gnh,
                geh: r.geh,
                gnw: r.gnw,
                gew: r.gew,
            };
        }
        SL_OK
    })
}

/// Interpolate an operational VMF grid at one point. The grid arrives as its own text with a
/// length, the way RINEX does, because it is a file rather than a token and this library does not
/// read files.
///
/// # Safety
/// `grid_text` is readable for `text_len` bytes; `out` is writable for one `sl_vmf_coefficients`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_vmf_grid_coefficients(
    grid_text: *const c_char,
    text_len: usize,
    latitude_rad: f64,
    longitude_rad: f64,
    out: *mut SlVmfCoefficients,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_vmf_grid_coefficients";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out pointer"));
        }
        // SAFETY: forwarding the caller's own contract.
        let Some(text) = (unsafe { borrow_sized_str(grid_text, text_len) }) else {
            return fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{WHO}: the grid text is null or is not UTF-8"),
            );
        };
        let grid = match VmfGrid::parse(text) {
            Ok(grid) => grid,
            Err(e) => return fail(SL_ERR_PARSE, format!("{WHO}: {e}")),
        };
        let c = grid.interpolate(latitude_rad, longitude_rad);
        // SAFETY: checked non-null.
        unsafe {
            *out = SlVmfCoefficients {
                ah: c.ah,
                aw: c.aw,
                zhd: c.zhd,
                zwd: c.zwd,
            };
        }
        SL_OK
    })
}

// ── Cloud masking (ONNX) ──────────────────────────────────────────────────────────────────────

/// `sl_cloud_masker`: a loaded cloud-segmentation model and the probability cut it thresholds at.
///
/// The field is behind the feature, so without `onnx` the type is a zero-sized shell no entry point
/// ever constructs — `sl_cloud_masker_open` refuses before it would.
pub struct SlCloudMasker {
    #[cfg(feature = "onnx")]
    masker: CloudMasker,
    #[cfg(not(feature = "onnx"))]
    _unbuilt: [u8; 0],
}

/// What a model has to be before it is loaded: some bytes, and a cut that is a number.
///
/// The cut is deliberately *not* held to `[0, 1]`. A threshold above every probability is how a
/// caller asks for "nothing is cloudy" and below every one for "everything is", and both are useful
/// answers rather than mistakes.
fn check_cloud_model(who: &str, onnx_len: usize, threshold: f32) -> Result<(), i32> {
    if onnx_len == 0 {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: the model is zero bytes long"),
        ));
    }
    if !threshold.is_finite() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: threshold is {threshold}, and must be finite"),
        ));
    }
    Ok(())
}

/// The tile length `channels x height x width` implies, or the refusal that says why there is none.
fn check_cloud_tile(who: &str, channels: usize, height: usize, width: usize) -> Result<usize, i32> {
    for (name, value) in [("channels", channels), ("height", height), ("width", width)] {
        if value == 0 {
            return Err(fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{who}: {name} is zero, and a tile has no pixels"),
            ));
        }
    }
    channels
        .checked_mul(height)
        .and_then(|hw| hw.checked_mul(width))
        .ok_or_else(|| {
            fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{who}: {channels}x{height}x{width} overflows a size_t"),
            )
        })
}

/// Load a model into a handle, or say that this build has nothing to load it with.
///
/// One function with two bodies rather than two `#[cfg]`-selected functions, because `cargo mutants`
/// emits a mutant for each arm of such a pair and a run can only judge the arm its configuration
/// compiles: the other is reported as a survivor that no test could have reached. `run_cloud_mask`
/// below is still a pair — nothing is emitted for the body of an `unsafe fn`, so it has no phantom
/// to avoid.
fn open_masker(who: &str, onnx_bytes: &[u8], threshold: f32) -> Result<SlCloudMasker, i32> {
    #[cfg(feature = "onnx")]
    match CloudMasker::new(onnx_bytes, threshold) {
        Ok(masker) => Ok(SlCloudMasker { masker }),
        Err(e) => Err(fail(SL_ERR_PARSE, format!("{who}: {e}"))),
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (onnx_bytes, threshold);
        Err(fail(
            SL_ERR_UNSUPPORTED,
            format!("{who}: this library was built without the `onnx` feature"),
        ))
    }
}

/// The mask a live handle produces for one tile, as `1 = cloudy` bytes.
///
/// Takes the handle as a raw pointer rather than a reference so that the build without `onnx` can
/// answer without dereferencing it: no entry point there can have produced a live handle, so a
/// non-null one is a caller's mistake and reading it would be the wrong way to find that out.
#[cfg(feature = "onnx")]
unsafe fn run_cloud_mask(
    who: &str,
    masker: *const SlCloudMasker,
    tile: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<u8>, i32> {
    // SAFETY: the caller checked it non-null and states the handle is live.
    let masker = unsafe { &*masker };
    match masker.masker.mask(tile, channels, height, width) {
        Ok(mask) => Ok(mask.into_iter().map(u8::from).collect()),
        Err(e) => Err(fail(SL_ERR_COMPUTE, format!("{who}: {e}"))),
    }
}

#[cfg(not(feature = "onnx"))]
unsafe fn run_cloud_mask(
    who: &str,
    _masker: *const SlCloudMasker,
    _tile: &[f32],
    _channels: usize,
    _height: usize,
    _width: usize,
) -> Result<Vec<u8>, i32> {
    Err(fail(
        SL_ERR_UNSUPPORTED,
        format!("{who}: this library was built without the `onnx` feature"),
    ))
}

/// Load a cloud-segmentation model into a handle. See the header.
///
/// # Safety
/// `onnx_bytes` is readable for `onnx_len` bytes; `out` is writable for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_cloud_masker_open(
    onnx_bytes: *const u8,
    onnx_len: usize,
    threshold: f32,
    out: *mut *mut SlCloudMasker,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_cloud_masker_open";
        if onnx_bytes.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_cloud_model(WHO, onnx_len, threshold) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is readable for `onnx_len` bytes.
        let bytes = unsafe { core::slice::from_raw_parts(onnx_bytes, onnx_len) };
        match open_masker(WHO, bytes, threshold) {
            Ok(masker) => {
                let handle = Box::into_raw(Box::new(masker));
                // SAFETY: checked non-null; written only now that the handle exists.
                unsafe { *out = handle };
                SL_OK
            }
            Err(status) => status,
        }
    })
}

/// Release a handle from [`sl_cloud_masker_open`].
///
/// # Safety
/// `masker` is null, or a handle from `sl_cloud_masker_open` that has not already been closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_cloud_masker_close(masker: *mut SlCloudMasker) {
    if masker.is_null() {
        return;
    }
    // SAFETY: the caller states this came from `Box::into_raw` in `open` and is unclosed.
    drop(unsafe { Box::from_raw(masker) });
}

/// Run a loaded model over one `channels x height x width` tile. See the header.
///
/// # Safety
/// `masker` is a live handle; `tile` is readable for `channels * height * width` floats; `out` is
/// null or writable for `cap` bytes; `out_len` is writable for one `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_cloud_masker_mask(
    masker: *const SlCloudMasker,
    tile: *const f32,
    channels: usize,
    height: usize,
    width: usize,
    out: *mut u8,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_cloud_masker_mask";
        if masker.is_null() || tile.is_null() || out_len.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let len = match check_cloud_tile(WHO, channels, height, width) {
            Ok(len) => len,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states it is readable for that many floats.
        let tile = unsafe { core::slice::from_raw_parts(tile, len) };
        // SAFETY: `masker` was checked non-null and the caller states the handle is live.
        match unsafe { run_cloud_mask(WHO, masker, tile, channels, height, width) } {
            // SAFETY: forwarding the caller's own contract for `out`, `cap` and `out_len`.
            Ok(mask) => unsafe { deliver(&mask, out, cap, out_len) },
            Err(status) => status,
        }
    })
}

/// Load a model and mask one tile in a single call. See the header.
///
/// # Safety
/// `onnx_bytes` is readable for `onnx_len` bytes; `tile` is readable for
/// `channels * height * width` floats; `out` is null or writable for `cap` bytes; `out_len` is
/// writable for one `usize`.
#[allow(clippy::too_many_arguments)] // the model, its cut, the tile's three extents and the buffer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_cloud_mask(
    onnx_bytes: *const u8,
    onnx_len: usize,
    threshold: f32,
    tile: *const f32,
    channels: usize,
    height: usize,
    width: usize,
    out: *mut u8,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    // No `guard` of its own: the two calls below each carry one, and nothing between them can
    // panic. Wrapping again would catch a panic that has already been turned into a status.
    let mut masker: *mut SlCloudMasker = core::ptr::null_mut();
    // SAFETY: forwarding the caller's contract for the model bytes; `masker` is a local pointer.
    let status = unsafe { sl_cloud_masker_open(onnx_bytes, onnx_len, threshold, &raw mut masker) };
    if status != SL_OK {
        return status;
    }
    // SAFETY: the handle is the one just opened, and the tile and buffer are the caller's.
    let status =
        unsafe { sl_cloud_masker_mask(masker, tile, channels, height, width, out, cap, out_len) };
    // SAFETY: the handle came from `sl_cloud_masker_open` above and has not been closed.
    unsafe { sl_cloud_masker_close(masker) };
    status
}

// ── Radiometric and multistatic observables ───────────────────────────────────────────────────
//
// The observables #314 and #324 added, in the shape a tracking system already holds them: a
// terminal is a position, a velocity and the epoch the signal was at it, and a link is two or three
// of those. Every geometry here can be refused for one reason — the epochs along a signal path have
// to increase — so the fallible ones are a status plus an out-parameter rather than a bare
// `double`, and only the two that cannot fail are total.

/// `sl_terminal`: one end of a link, and when the signal was there.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlTerminal {
    /// Position in the frame the satellite state is in, metres.
    pub position: SlVec3,
    /// Velocity in that frame, m/s.
    pub velocity: SlVec3,
    /// The epoch the signal left or reached this terminal, seconds.
    pub epoch: f64,
}

/// `sl_band`: Moyer's three DSN bands, which fix the turn-around ratios.
const BAND_S: i32 = 0;
const BAND_X: i32 = 1;
const BAND_KA: i32 = 2;

/// `sl_ramp_segment`: one linear ramp of a station's exciter.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlRampSegment {
    /// Start of the segment's validity, seconds.
    pub start: f64,
    /// End of it, seconds.
    pub end: f64,
    /// The transmitted frequency at `start`, hertz.
    pub start_frequency: f64,
    /// The ramp rate over the segment, hertz per second.
    pub rate: f64,
}

/// `sl_doppler_link`: what an n-way averaged Doppler count is taken against.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlDopplerLink {
    /// The band transmitted on, an `sl_band`.
    pub uplink: i32,
    /// The band received on, an `sl_band`.
    pub downlink: i32,
    /// The receiver's reference frequency, hertz.
    pub reference_frequency: f64,
    /// The band that reference is stated against, an `sl_band`.
    pub reference_band: i32,
    /// The count interval the average is taken over, seconds.
    pub count_interval: f64,
    /// Whether the ramp term enters negative, which is the DSN convention.
    pub subtracted: bool,
}

/// `sl_corona_term`: one term of the inverse-power electron-density series.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlCoronaTerm {
    /// The coefficient, electrons per cubic metre.
    pub coefficient: f64,
    /// The exponent `k` of `(r/R_sun)^-k`.
    pub exponent: f64,
}

fn to_terminal(t: SlTerminal) -> Terminal {
    Terminal::new(
        Station {
            position: to_three(t.position),
            velocity: to_three(t.velocity),
        },
        t.epoch,
    )
}

/// An `sl_band` tag as the band it names, or a refusal naming the value that is not one.
fn to_band(who: &str, name: &str, tag: i32) -> Result<Band, i32> {
    match tag {
        BAND_S => Ok(Band::S),
        BAND_X => Ok(Band::X),
        BAND_KA => Ok(Band::Ka),
        other => Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: {name} is {other}, which is not an sl_band"),
        )),
    }
}

/// The geometry both bistatic observables take, so the epoch ordering is checked in one place.
fn to_bistatic(
    who: &str,
    transmitter: SlTerminal,
    bounce: f64,
    receiver: SlTerminal,
) -> Result<BistaticGeometry, i32> {
    BistaticGeometry::new(to_terminal(transmitter), bounce, to_terminal(receiver))
        .map_err(|e| fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: {e}")))
}

/// The geometry TDOA, FDOA and ΔDOR share.
fn to_tdoa(
    who: &str,
    emission: f64,
    first: SlTerminal,
    second: SlTerminal,
) -> Result<TdoaGeometry, i32> {
    TdoaGeometry::new(emission, to_terminal(first), to_terminal(second))
        .map_err(|e| fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: {e}")))
}

/// Read the three pointers every bistatic entry point takes, or say which one was null.
///
/// # Safety
/// Each pointer is null or readable for one value of its type.
unsafe fn read_bistatic(
    who: &str,
    satellite: *const SlState,
    transmitter: *const SlTerminal,
    receiver: *const SlTerminal,
    out: *mut f64,
) -> Result<([f64; 6], SlTerminal, SlTerminal), i32> {
    if satellite.is_null() || transmitter.is_null() || receiver.is_null() || out.is_null() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: null pointer"),
        ));
    }
    // SAFETY: each was checked non-null just above, and the caller states each is readable.
    unsafe { Ok((to_six(*satellite), *transmitter, *receiver)) }
}

/// Bistatic range `rho_tx + rho_rx`, metres.
///
/// # Safety
/// `satellite`, `transmitter` and `receiver` each point at one initialised value; `out` is writable
/// for one double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_bistatic_range(
    satellite: *const SlState,
    transmitter: *const SlTerminal,
    bounce: f64,
    receiver: *const SlTerminal,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_bistatic_range";
        // SAFETY: forwarding the caller's contract on the four pointers.
        let (state, tx, rx) =
            match unsafe { read_bistatic(WHO, satellite, transmitter, receiver, out) } {
                Ok(read) => read,
                Err(status) => return status,
            };
        let geometry = match to_bistatic(WHO, tx, bounce, rx) {
            Ok(geometry) => geometry,
            Err(status) => return status,
        };
        // SAFETY: `out` was checked non-null above.
        unsafe { *out = estimation_bistatic_range(&state, &geometry) };
        SL_OK
    })
}

/// Bistatic range rate, m/s — the time derivative of `sl_bistatic_range`.
///
/// # Safety
/// As `sl_bistatic_range`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_bistatic_range_rate(
    satellite: *const SlState,
    transmitter: *const SlTerminal,
    bounce: f64,
    receiver: *const SlTerminal,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_bistatic_range_rate";
        // SAFETY: forwarding the caller's contract on the four pointers.
        let (state, tx, rx) =
            match unsafe { read_bistatic(WHO, satellite, transmitter, receiver, out) } {
                Ok(read) => read,
                Err(status) => return status,
            };
        let geometry = match to_bistatic(WHO, tx, bounce, rx) {
            Ok(geometry) => geometry,
            Err(status) => return status,
        };
        // SAFETY: `out` was checked non-null above.
        unsafe { *out = estimation_bistatic_range_rate(&state, &geometry) };
        SL_OK
    })
}

/// Time difference of arrival `(rho_second - rho_first)/c`, seconds.
///
/// # Safety
/// `satellite`, `first` and `second` each point at one initialised value; `out` is writable for one
/// double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_tdoa(
    satellite: *const SlState,
    emission: f64,
    first: *const SlTerminal,
    second: *const SlTerminal,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_tdoa";
        // SAFETY: forwarding the caller's contract; the reader takes terminals in either role.
        let (state, a, b) = match unsafe { read_bistatic(WHO, satellite, first, second, out) } {
            Ok(read) => read,
            Err(status) => return status,
        };
        let geometry = match to_tdoa(WHO, emission, a, b) {
            Ok(geometry) => geometry,
            Err(status) => return status,
        };
        // SAFETY: `out` was checked non-null above.
        unsafe { *out = estimation_tdoa(&state, &geometry) };
        SL_OK
    })
}

/// Frequency difference of arrival at carrier `frequency`, hertz.
///
/// # Safety
/// As `sl_tdoa`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_fdoa(
    satellite: *const SlState,
    emission: f64,
    first: *const SlTerminal,
    second: *const SlTerminal,
    frequency: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_fdoa";
        // SAFETY: forwarding the caller's contract; the reader takes terminals in either role.
        let (state, a, b) = match unsafe { read_bistatic(WHO, satellite, first, second, out) } {
            Ok(read) => read,
            Err(status) => return status,
        };
        let geometry = match to_tdoa(WHO, emission, a, b) {
            Ok(geometry) => geometry,
            Err(status) => return status,
        };
        // SAFETY: `out` was checked non-null above.
        unsafe { *out = estimation_fdoa(&state, &geometry, frequency) };
        SL_OK
    })
}

/// ΔDOR against a quasar at `right_ascension`/`declination`, seconds.
///
/// # Safety
/// As `sl_tdoa`.
#[allow(clippy::too_many_arguments)] // the state, the emission, two terminals, two source angles
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_delta_dor(
    satellite: *const SlState,
    emission: f64,
    first: *const SlTerminal,
    second: *const SlTerminal,
    right_ascension: f64,
    declination: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_delta_dor";
        // SAFETY: forwarding the caller's contract; the reader takes terminals in either role.
        let (state, a, b) = match unsafe { read_bistatic(WHO, satellite, first, second, out) } {
            Ok(read) => read,
            Err(status) => return status,
        };
        let geometry = match to_tdoa(WHO, emission, a, b) {
            Ok(geometry) => geometry,
            Err(status) => return status,
        };
        let quasar = Quasar {
            right_ascension,
            declination,
        };
        // SAFETY: `out` was checked non-null above.
        unsafe { *out = estimation_delta_dor(&state, &geometry, quasar) };
        SL_OK
    })
}

/// Separation of two satellites, metres. Total.
///
/// # Safety
/// Both pointers are null or point at one initialised `sl_state`; null answers NaN.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_inter_satellite_range(
    local: *const SlState,
    remote: *const SlState,
) -> f64 {
    guard_f64(|| {
        if local.is_null() || remote.is_null() {
            return f64::NAN;
        }
        // SAFETY: checked non-null just above.
        let (a, b) = unsafe { (to_six(*local), to_six(*remote)) };
        estimation_inter_satellite_range(&a, &b)
    })
}

/// Inter-satellite range rate, m/s — positive when the two are separating. Total.
///
/// # Safety
/// As `sl_inter_satellite_range`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_inter_satellite_range_rate(
    local: *const SlState,
    remote: *const SlState,
) -> f64 {
    guard_f64(|| {
        if local.is_null() || remote.is_null() {
            return f64::NAN;
        }
        // SAFETY: checked non-null just above.
        let (a, b) = unsafe { (to_six(*local), to_six(*remote)) };
        estimation_inter_satellite_range_rate(&a, &b)
    })
}

/// The Shapiro delay on one leg, metres. Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_shapiro_delay(r_emitter: f64, r_receiver: f64, path: f64, gm: f64) -> f64 {
    guard_f64(|| estimation_shapiro_delay(r_emitter, r_receiver, path, gm))
}

/// Borrow a station's ramp table, or say why it is not one.
///
/// # Safety
/// `segments` is null or readable for `segment_count` `sl_ramp_segment`.
unsafe fn borrow_ramps(
    who: &str,
    segments: *const SlRampSegment,
    segment_count: usize,
) -> Result<Vec<RampSegment>, i32> {
    if segments.is_null() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: segments is null"),
        ));
    }
    check_nonzero(who, "segment_count", segment_count)?;
    // SAFETY: checked non-null; the caller states it is readable for `segment_count`.
    let raw = unsafe { core::slice::from_raw_parts(segments, segment_count) };
    Ok(raw
        .iter()
        .map(|s| RampSegment {
            start: s.start,
            end: s.end,
            start_frequency: s.start_frequency,
            rate: s.rate,
        })
        .collect())
}

/// DSN sequential range in range units, wrapped at `2^(C+6)`.
///
/// # Safety
/// `segments` is readable for `segment_count` `sl_ramp_segment`; `out` is writable for one double.
#[allow(clippy::too_many_arguments)] // the table and its length, the band, the interval, and `C`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_dsn_sequential_range(
    segments: *const SlRampSegment,
    segment_count: usize,
    uplink: i32,
    transmit: f64,
    receive: f64,
    lowest_ranging_component: i32,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_dsn_sequential_range";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out"));
        }
        // SAFETY: forwarding the caller's contract on `segments` and `segment_count`.
        let owned = match unsafe { borrow_ramps(WHO, segments, segment_count) } {
            Ok(owned) => owned,
            Err(status) => return status,
        };
        let band = match to_band(WHO, "uplink", uplink) {
            Ok(band) => band,
            Err(status) => return status,
        };
        let table = match RampTable::new(&owned) {
            Ok(table) => table,
            Err(e) => return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: {e}")),
        };
        match estimation_sequential_range(&table, band, transmit, receive, lowest_ranging_component)
        {
            // SAFETY: `out` was checked non-null above.
            Ok(value) => unsafe {
                *out = value;
                SL_OK
            },
            Err(e) => fail(SL_ERR_DOMAIN, format!("{WHO}: {e}")),
        }
    })
}

/// DSN n-way averaged Doppler between two transmission epochs, hertz.
///
/// # Safety
/// `segments` is readable for `segment_count` `sl_ramp_segment`; `link` points at one initialised
/// `sl_doppler_link`; `out` is writable for one double.
#[allow(clippy::too_many_arguments)] // the table and its length, the link, and the two epochs
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_dsn_averaged_doppler(
    segments: *const SlRampSegment,
    segment_count: usize,
    link: *const SlDopplerLink,
    transmit_start: f64,
    transmit_end: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_dsn_averaged_doppler";
        if link.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: forwarding the caller's contract on `segments` and `segment_count`.
        let owned = match unsafe { borrow_ramps(WHO, segments, segment_count) } {
            Ok(owned) => owned,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states it is initialised.
        let l = unsafe { *link };
        let bands = [
            ("uplink", l.uplink),
            ("downlink", l.downlink),
            ("reference_band", l.reference_band),
        ]
        .map(|(name, tag)| to_band(WHO, name, tag));
        let [uplink, downlink, reference] = bands;
        let (uplink, downlink, reference) = match (uplink, downlink, reference) {
            (Ok(u), Ok(d), Ok(r)) => (u, d, r),
            (Err(status), _, _) | (_, Err(status), _) | (_, _, Err(status)) => return status,
        };
        let table = match RampTable::new(&owned) {
            Ok(table) => table,
            Err(e) => return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: {e}")),
        };
        let doppler = DopplerLink {
            uplink,
            downlink,
            reference: DopplerReference {
                frequency: l.reference_frequency,
                band: reference,
            },
            signature: if l.subtracted {
                DopplerSignature::Subtracted
            } else {
                DopplerSignature::Added
            },
            count_interval: l.count_interval,
        };
        match estimation_averaged_doppler(&table, doppler, transmit_start, transmit_end) {
            // SAFETY: `out` was checked non-null above.
            Ok(value) => unsafe {
                *out = value;
                SL_OK
            },
            Err(e) => fail(SL_ERR_DOMAIN, format!("{WHO}: {e}")),
        }
    })
}

/// DSN measured frequency, hertz — the instantaneous counterpart of the averaged Doppler.
///
/// # Safety
/// `segments` is readable for `segment_count` `sl_ramp_segment`; `out` is writable for one double.
#[allow(clippy::too_many_arguments)] // the table and its length, two bands, the epoch and the rate
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_dsn_measured_frequency(
    segments: *const SlRampSegment,
    segment_count: usize,
    uplink: i32,
    downlink: i32,
    transmit_epoch: f64,
    transmit_rate: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_dsn_measured_frequency";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out"));
        }
        // SAFETY: forwarding the caller's contract on `segments` and `segment_count`.
        let owned = match unsafe { borrow_ramps(WHO, segments, segment_count) } {
            Ok(owned) => owned,
            Err(status) => return status,
        };
        let (up, down) = match (
            to_band(WHO, "uplink", uplink),
            to_band(WHO, "downlink", downlink),
        ) {
            (Ok(up), Ok(down)) => (up, down),
            (Err(status), _) | (_, Err(status)) => return status,
        };
        let table = match RampTable::new(&owned) {
            Ok(table) => table,
            Err(e) => return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: {e}")),
        };
        match estimation_measured_frequency(&table, up, down, transmit_epoch, transmit_rate) {
            // SAFETY: `out` was checked non-null above.
            Ok(value) => unsafe {
                *out = value;
                SL_OK
            },
            Err(e) => fail(SL_ERR_DOMAIN, format!("{WHO}: {e}")),
        }
    })
}

/// Solar-corona range correction along one leg, metres.
///
/// `series` may be null, which takes Aksim & Pavlov (2022).
///
/// # Safety
/// Both position pointers point at one `sl_vec3`; `series` is null or readable for `series_len`
/// `sl_corona_term`; `out` is writable for one double.
#[allow(clippy::too_many_arguments)] // two endpoints, the frequency, the sign, the series, the out
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_solar_corona_delay(
    transmitter_from_sun: *const SlVec3,
    receiver_from_sun: *const SlVec3,
    frequency: f64,
    group_delay: bool,
    series: *const SlCoronaTerm,
    series_len: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_solar_corona_delay";
        if transmitter_from_sun.is_null() || receiver_from_sun.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: checked non-null just above.
        let (tx, rx) = unsafe {
            (
                to_three(*transmitter_from_sun),
                to_three(*receiver_from_sun),
            )
        };
        let owned: Vec<CoronaTerm> = if series.is_null() {
            AKSIM_PAVLOV_2022.to_vec()
        } else {
            if let Err(status) = check_nonzero(WHO, "series_len", series_len) {
                return status;
            }
            // SAFETY: checked non-null; the caller states it is readable for `series_len`.
            unsafe { core::slice::from_raw_parts(series, series_len) }
                .iter()
                .map(|t| CoronaTerm {
                    coefficient: t.coefficient,
                    exponent: t.exponent,
                })
                .collect()
        };
        let model = match CoronaSeries::new(&owned) {
            Ok(model) => model,
            Err(e) => return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: {e}")),
        };
        let signature = if group_delay {
            PlasmaSignature::GroupDelay
        } else {
            PlasmaSignature::PhaseAdvance
        };
        match estimation_solar_corona_delay(&model, &tx, &rx, frequency, signature) {
            // SAFETY: `out` was checked non-null above.
            Ok(value) => unsafe {
                *out = value;
                SL_OK
            },
            Err(e) => fail(SL_ERR_DOMAIN, format!("{WHO}: {e}")),
        }
    })
}

// ── Orbital lifetime, re-entry and element-set generation ─────────────────────────────────────
//
// #310's decay scan and #311's TLE printer. The scan takes thirteen numbers that are all one
// decision — the field, the spacecraft, the atmosphere and how finely to look — so they arrive as a
// struct with an `_init` beside it, the way `sl_dsst_params` and `sl_walker_params` do. The zonal
// list is the one variable-length input and stays a pointer and a count.

/// `sl_lifetime_outcome`, as the library matches on it.
const LIFETIME_REENTRY: i32 = 0;
const LIFETIME_SURVIVES: i32 = 1;
const LIFETIME_UNRESOLVED: i32 = 2;

/// `sl_decay_params`: the gravity field, the spacecraft, the atmosphere and the scan.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlDecayParams {
    /// Drag coefficient, dimensionless.
    pub cd: f64,
    /// Area-to-mass ratio, m²/kg.
    pub area_over_mass: f64,
    /// Density at `reference_radius`, kg/m³.
    pub reference_density: f64,
    /// The spherical radius the reference density is quoted at, metres.
    pub reference_radius: f64,
    /// Density scale height, metres.
    pub scale_height: f64,
    /// How far ahead to look, seconds.
    pub horizon: f64,
    /// How far the scan advances between perigee tests, seconds.
    pub chunk: f64,
    /// How finely the crossing chunk is bisected, seconds.
    pub tolerance: f64,
    /// A multiplier on the density; this is the ensemble's own axis.
    pub density_scale: f64,
    /// Gravitational parameter, m³/s².
    pub mu: f64,
    /// Body equatorial radius, metres.
    pub body_radius: f64,
    /// The mean perigee altitude counted as re-entry, metres.
    pub reentry_altitude: f64,
    /// Atmosphere rotation rate about `z`, rad/s.
    pub rotation_rate: f64,
}

/// `sl_lifetime`: what became of one object over one horizon.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlLifetime {
    /// Which of the three answers this is, an `sl_lifetime_outcome`.
    pub outcome: i32,
    /// Seconds to the crossing, to the end of the horizon, or to where the scan stopped, according
    /// to `outcome`.
    pub seconds: f64,
    /// The mean perigee altitude at that epoch, metres.
    pub perigee_altitude_m: f64,
}

/// `sl_lifetime_ensemble`: how many members an ensemble had and how they ended.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlLifetimeEnsemble {
    /// Members run.
    pub members: usize,
    /// Members still up at the horizon.
    pub survivors: usize,
    /// Members whose propagation stopped above the threshold.
    pub unresolved: usize,
}

/// Fill `p` with the defaults documented in the header.
///
/// # Safety
/// `p` is null or writable for one `sl_decay_params`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_decay_params_init(p: *mut SlDecayParams) {
    if p.is_null() {
        return;
    }
    // SAFETY: checked non-null just above.
    unsafe {
        *p = SlDecayParams {
            cd: 2.2,
            area_over_mass: 0.01,
            reference_density: f64::NAN,
            reference_radius: f64::NAN,
            scale_height: f64::NAN,
            horizon: f64::NAN,
            chunk: 86_400.0,
            tolerance: 60.0,
            density_scale: 1.0,
            mu: MU_EARTH,
            body_radius: RE_EARTH,
            reentry_altitude: DEFAULT_REENTRY_ALTITUDE,
            rotation_rate: spacelab_frames::EARTH_ROTATION_RATE,
        };
    }
}

/// An exponential atmosphere about the spherical radius, co-rotating rigidly about `z`.
///
/// The one atmosphere this slice offers, and named rather than implied: driving the mean-element
/// quadrature with an empirical model over a decade needs the inertial-to-geodetic rotation carried
/// through every node, which is a separate piece of work. `scale` is how an ensemble member differs
/// from its neighbours.
struct CapiDecayAtmosphere {
    reference_density: f64,
    reference_radius: f64,
    scale_height: f64,
    omega: f64,
    scale: f64,
}

impl DecayAtmosphere for CapiDecayAtmosphere {
    fn density(&self, _t: f64, position: [f64; 3]) -> f64 {
        let radius = (position[0].mul_add(position[0], position[1] * position[1])
            + position[2] * position[2])
            .sqrt();
        self.scale
            * self.reference_density
            * ((self.reference_radius - radius) / self.scale_height).exp()
    }

    fn velocity(&self, position: [f64; 3]) -> [f64; 3] {
        [-self.omega * position[1], self.omega * position[0], 0.0]
    }
}

/// The zonal field and the cannonball, owned so a `DecayModel` can borrow them.
struct CapiDecay {
    zonal: DsstZonal,
    drag: DsstAtmosphericDrag,
    mu: f64,
    body_radius: f64,
    reentry_altitude: f64,
}

impl CapiDecay {
    fn model(&self) -> DecayModel<'_> {
        DecayModel {
            zonal: &self.zonal,
            drag: self.drag,
            mu: self.mu,
            body_radius: self.body_radius,
            reentry_altitude: self.reentry_altitude,
        }
    }
}

/// The decay model and the atmosphere, validated once for all three lifetime entry points.
///
/// # Safety
/// `params` and `initial` are non-null and readable for their documented extents; `zonals` is
/// readable for `zonal_count` doubles.
unsafe fn read_decay(
    who: &str,
    params: *const SlDecayParams,
    initial: *const f64,
    zonals: *const f64,
    zonal_count: usize,
) -> Result<(SlDecayParams, [f64; 6], CapiDecay), i32> {
    if params.is_null() || initial.is_null() || zonals.is_null() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: null pointer"),
        ));
    }
    // `DsstZonal::new` asserts on a short list, and an assertion across the ABI is an abort rather
    // than a status, so the length is checked here, where it can still be reported.
    check_at_least(who, "zonal_count", zonal_count, 3)?;
    // SAFETY: checked non-null; the caller states the extents.
    let (p, state, zonal_slice) = unsafe {
        (
            *params,
            {
                let mut s = [0.0_f64; 6];
                s.copy_from_slice(core::slice::from_raw_parts(initial, 6));
                s
            },
            core::slice::from_raw_parts(zonals, zonal_count),
        )
    };
    check_all_finite(who, "the initial state", &state)?;
    check_all_finite(who, "zonals", zonal_slice)?;
    for (name, value) in [
        ("cd", p.cd),
        ("area_over_mass", p.area_over_mass),
        ("reference_density", p.reference_density),
        ("reference_radius", p.reference_radius),
        ("scale_height", p.scale_height),
        ("horizon", p.horizon),
        ("chunk", p.chunk),
        ("tolerance", p.tolerance),
        ("density_scale", p.density_scale),
        ("mu", p.mu),
        ("body_radius", p.body_radius),
    ] {
        check_positive(who, name, value, SL_ERR_INVALID_ARGUMENT)?;
    }
    check_finite(who, "reentry_altitude", p.reentry_altitude)?;
    check_finite(who, "rotation_rate", p.rotation_rate)?;
    Ok((
        p,
        state,
        CapiDecay {
            zonal: DsstZonal::new(p.mu, p.body_radius, zonal_slice.to_vec()),
            drag: DsstAtmosphericDrag::new(p.cd, p.area_over_mass, p.mu),
            mu: p.mu,
            body_radius: p.body_radius,
            reentry_altitude: p.reentry_altitude,
        },
    ))
}

/// The atmosphere one member flies through.
fn decay_atmosphere(p: SlDecayParams, scale: f64) -> CapiDecayAtmosphere {
    CapiDecayAtmosphere {
        reference_density: p.reference_density,
        reference_radius: p.reference_radius,
        scale_height: p.scale_height,
        omega: p.rotation_rate,
        scale,
    }
}

/// The three named outcomes, flattened without merging them.
fn lifetime_out(outcome: Lifetime) -> SlLifetime {
    match outcome {
        Lifetime::Reentry {
            seconds,
            perigee_altitude,
        } => SlLifetime {
            outcome: LIFETIME_REENTRY,
            seconds,
            perigee_altitude_m: perigee_altitude,
        },
        Lifetime::Survives {
            horizon,
            perigee_altitude,
        } => SlLifetime {
            outcome: LIFETIME_SURVIVES,
            seconds: horizon,
            perigee_altitude_m: perigee_altitude,
        },
        Lifetime::Unresolved {
            seconds,
            perigee_altitude,
        } => SlLifetime {
            outcome: LIFETIME_UNRESOLVED,
            seconds,
            perigee_altitude_m: perigee_altitude,
        },
    }
}

/// Predict one object's re-entry epoch under drag and a zonal field.
///
/// # Safety
/// `params` points at one initialised `sl_decay_params`; `initial` is readable for six doubles;
/// `zonals` is readable for `zonal_count` doubles; `out` is writable for one `sl_lifetime`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_orbital_lifetime(
    params: *const SlDecayParams,
    initial: *const f64,
    zonals: *const f64,
    zonal_count: usize,
    out: *mut SlLifetime,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_orbital_lifetime";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out"));
        }
        // SAFETY: forwarding the caller's contract on the three input pointers.
        let (p, state, decay) =
            match unsafe { read_decay(WHO, params, initial, zonals, zonal_count) } {
                Ok(read) => read,
                Err(status) => return status,
            };
        match prop_lifetime(
            &decay.model(),
            &decay_atmosphere(p, p.density_scale),
            state,
            p.horizon,
            p.chunk,
            p.tolerance,
        ) {
            // SAFETY: `out` was checked non-null above.
            Ok(outcome) => unsafe {
                *out = lifetime_out(outcome);
                SL_OK
            },
            Err(e) => fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: {e}")),
        }
    })
}

/// The same decay under a set of density scale factors, reported as quantiles.
///
/// A quantile the sample does not determine — past the censoring fraction, or from an empty
/// ensemble — is written as NaN, because the alternative is the largest decay the run happened to
/// see, which understates the answer by however long the censored members go on to last.
///
/// # Safety
/// As `sl_orbital_lifetime`, plus: `density_scales` is readable for `scale_count` doubles,
/// `quantiles` for `quantile_count`, and `out_seconds` is writable for `quantile_count`.
#[allow(clippy::too_many_arguments)] // the scan's four inputs, the ensemble's two, and two outputs
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_orbital_lifetime_ensemble(
    params: *const SlDecayParams,
    initial: *const f64,
    zonals: *const f64,
    zonal_count: usize,
    density_scales: *const f64,
    scale_count: usize,
    quantiles: *const f64,
    quantile_count: usize,
    out_seconds: *mut f64,
    out: *mut SlLifetimeEnsemble,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_orbital_lifetime_ensemble";
        if density_scales.is_null() || quantiles.is_null() || out_seconds.is_null() || out.is_null()
        {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_nonzero(WHO, "scale_count", scale_count) {
            return status;
        }
        if let Err(status) = check_nonzero(WHO, "quantile_count", quantile_count) {
            return status;
        }
        // SAFETY: forwarding the caller's contract on the three input pointers.
        let (p, state, decay) =
            match unsafe { read_decay(WHO, params, initial, zonals, zonal_count) } {
                Ok(read) => read,
                Err(status) => return status,
            };
        // SAFETY: checked non-null; the caller states both extents.
        let (scales, ps) = unsafe {
            (
                core::slice::from_raw_parts(density_scales, scale_count),
                core::slice::from_raw_parts(quantiles, quantile_count),
            )
        };
        if let Err(status) = check_all_finite(WHO, "density_scales", scales) {
            return status;
        }
        let ensemble = match prop_ensemble_lifetime(
            &decay.model(),
            |member| decay_atmosphere(p, scales[member]),
            scale_count,
            state,
            p.horizon,
            p.chunk,
            p.tolerance,
        ) {
            Ok(ensemble) => ensemble,
            Err((member, e)) => {
                return fail(
                    SL_ERR_INVALID_ARGUMENT,
                    format!("{WHO}: member {member}: {e}"),
                );
            }
        };
        for (i, &q) in ps.iter().enumerate() {
            // SAFETY: `out_seconds` is writable for `quantile_count`, and `i` is below it.
            unsafe { *out_seconds.add(i) = ensemble.quantile(q).unwrap_or(f64::NAN) };
        }
        // SAFETY: `out` was checked non-null above.
        unsafe {
            *out = SlLifetimeEnsemble {
                members: ensemble.members(),
                survivors: ensemble.survivors(),
                unresolved: ensemble.unresolved(),
            };
        }
        SL_OK
    })
}

/// `d ln(lifetime) / d ln(S/m)` by central difference on the area-to-mass ratio.
///
/// # Safety
/// As `sl_orbital_lifetime`, with `out` writable for one double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ballistic_log_sensitivity(
    params: *const SlDecayParams,
    initial: *const f64,
    zonals: *const f64,
    zonal_count: usize,
    fraction: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_ballistic_log_sensitivity";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out"));
        }
        if let Err(status) = check_positive(WHO, "fraction", fraction, SL_ERR_INVALID_ARGUMENT) {
            return status;
        }
        // SAFETY: forwarding the caller's contract on the three input pointers.
        let (p, state, decay) =
            match unsafe { read_decay(WHO, params, initial, zonals, zonal_count) } {
                Ok(read) => read,
                Err(status) => return status,
            };
        match prop_ballistic_sensitivity(
            &decay.model(),
            &decay_atmosphere(p, p.density_scale),
            state,
            p.horizon,
            p.chunk,
            p.tolerance,
            fraction,
        ) {
            // SAFETY: `out` was checked non-null above.
            Ok(value) => unsafe {
                *out = value;
                SL_OK
            },
            Err(e) => fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: {e}")),
        }
    })
}

/// `sl_utc_datetime`: a civil epoch, which is what a TLE's columns carry.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlUtcDateTime {
    /// Four-digit year.
    pub year: i32,
    /// Month, 1-12.
    pub month: u8,
    /// Day of month, 1-31.
    pub day: u8,
    /// Hour, 0-23.
    pub hour: u8,
    /// Minute, 0-59.
    pub minute: u8,
    /// Second, including the fraction.
    pub second: f64,
}

/// Bytes each TLE line needs: 69 columns and the terminator.
const TLE_LINE_BYTES: usize = 70;

/// `sl_tle`: the two lines, each NUL-terminated.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlTle {
    /// Line 1, 69 columns then a NUL.
    pub line1: [c_char; TLE_LINE_BYTES],
    /// Line 2, 69 columns then a NUL.
    pub line2: [c_char; TLE_LINE_BYTES],
}

/// Copy one 69-column line into its fixed field, refusing anything that is not 69 columns.
fn write_tle_line(who: &str, text: &str, field: &mut [c_char; TLE_LINE_BYTES]) -> Result<(), i32> {
    if text.len() != TLE_LINE_BYTES - 1 {
        return Err(fail(
            SL_ERR_COMPUTE,
            format!(
                "{who}: the builder produced {} columns, not {}",
                text.len(),
                TLE_LINE_BYTES - 1
            ),
        ));
    }
    for (slot, byte) in field.iter_mut().zip(text.bytes()) {
        #[expect(
            clippy::cast_possible_wrap,
            reason = "a TLE line is ASCII, so every byte is below 0x80"
        )]
        {
            *slot = byte as c_char;
        }
    }
    field[TLE_LINE_BYTES - 1] = 0;
    Ok(())
}

/// Print an element set as the two lines of a TLE.
///
/// # Safety
/// `epoch` points at one `sl_utc_datetime`; `parameters` is readable for six doubles;
/// `international_designator` is null or a NUL-terminated string; `out` is writable for one
/// `sl_tle`.
#[allow(clippy::too_many_arguments)] // the epoch, the identity, six elements, `B*` and two counters
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_tle_lines(
    epoch: *const SlUtcDateTime,
    norad_id: u32,
    parameters: *const f64,
    drag_term: f64,
    international_designator: *const c_char,
    element_set_number: u64,
    revolution_number: u64,
    out: *mut SlTle,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_tle_lines";
        if epoch.is_null() || parameters.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: checked non-null; the caller states `parameters` is readable for six.
        let (e, fitted) = unsafe {
            let mut p = [0.0_f64; 6];
            p.copy_from_slice(core::slice::from_raw_parts(parameters, 6));
            (*epoch, p)
        };
        if let Err(status) = check_all_finite(WHO, "parameters", &fitted) {
            return status;
        }
        if let Err(status) = check_finite(WHO, "drag_term", drag_term) {
            return status;
        }
        let designator = if international_designator.is_null() {
            None
        } else {
            // SAFETY: the caller states it is NUL-terminated; the scan is bounded.
            match unsafe { borrow_c_str(international_designator) } {
                Some(text) => Some(text),
                None => {
                    return fail(
                        SL_ERR_INVALID_ARGUMENT,
                        format!("{WHO}: international_designator is not UTF-8 or is unterminated"),
                    );
                }
            }
        };
        let utc = match UtcDateTime::new(e.year, e.month, e.day, e.hour, e.minute, e.second) {
            Ok(utc) => utc,
            Err(err) => return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: {err}")),
        };
        let seeded = match seed_elements(utc, NoradId::new(norad_id), &fitted) {
            Ok(seeded) => seeded,
            Err(err) => return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: {err}")),
        };
        // Through `TleModel` rather than writing the fields back by hand: it is the same write-back
        // the fit uses, so a set printed here and one fitted there differ in their values and not in
        // their conventions -- including the reduction of every angle into `[0, 360)`.
        let model = TleModel::new(seeded, Convention::Reference);
        let mut with_drag = [0.0_f64; 7];
        with_drag[..6].copy_from_slice(&fitted);
        with_drag[6] = drag_term;
        let elements = model.elements_with_drag(&with_drag);
        let mut builder = TleBuilder::from_elements(&elements)
            .element_set_number(element_set_number)
            .revolution_number(revolution_number);
        if let Some(text) = designator {
            builder = builder.international_designator(text);
        }
        let tle = match builder.build() {
            Ok(tle) => tle,
            Err(err) => return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: {err}")),
        };
        let mut lines = SlTle {
            line1: [0; TLE_LINE_BYTES],
            line2: [0; TLE_LINE_BYTES],
        };
        if let Err(status) = write_tle_line(WHO, tle.line1(), &mut lines.line1) {
            return status;
        }
        if let Err(status) = write_tle_line(WHO, tle.line2(), &mut lines.line2) {
            return status;
        }
        // SAFETY: `out` was checked non-null above.
        unsafe { *out = lines };
        SL_OK
    })
}

// ── GNSS observables: combinations, cycle slips and ambiguity resolution ──────────────────────
//
// The measurement side #312 added. The two detectors are stateful in Rust and batch here, for the
// same reason the Python bridge batches them: an arc is the unit a verdict is read on, and one
// crossing per sample would cost the crossing and buy nothing.

/// `sl_lanes`: the four wavelengths of a frequency pair, metres.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlLanes {
    /// `c/f1`.
    pub lambda1: f64,
    /// `c/f2`.
    pub lambda2: f64,
    /// `c/(f1 - f2)`.
    pub wide_lane: f64,
    /// `c/(f1 + f2)`.
    pub narrow_lane: f64,
}

/// `sl_slip`, as the library matches on it.
const SLIP_ARC_START: i32 = 0;
const SLIP_CONTINUOUS: i32 = 1;
const SLIP_SLIPPED: i32 = 2;

/// `sl_slip_verdict`: what a detector made of one sample.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlSlipVerdict {
    /// Which verdict, an `sl_slip`.
    pub verdict: i32,
    /// The test statistic, or NaN at an arc start, where there is none.
    pub statistic: f64,
}

/// The carrier frequency of `band` on constellation `system`, hertz, or NaN where that
/// constellation does not broadcast it. Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_carrier_frequency(system: c_char, band: c_char) -> f64 {
    guard_f64(|| {
        let ascii = |c: c_char| u8::try_from(c).ok().map(char::from);
        match (ascii(system), ascii(band)) {
            (Some(s), Some(b)) => gnss_carrier_frequency(s, b).unwrap_or(f64::NAN),
            _ => f64::NAN,
        }
    })
}

/// The pair every dual-frequency combination is formed on, or a refusal naming the frequency.
fn to_pair(who: &str, f1: f64, f2: f64) -> Result<FrequencyPair, i32> {
    FrequencyPair::new(f1, f2).map_err(|e| fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: {e}")))
}

/// The four lane wavelengths of a frequency pair.
///
/// # Safety
/// `out` is writable for one `sl_lanes`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_lane_wavelengths(f1: f64, f2: f64, out: *mut SlLanes) -> i32 {
    guard(|| {
        const WHO: &str = "sl_lane_wavelengths";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out"));
        }
        let pair = match to_pair(WHO, f1, f2) {
            Ok(pair) => pair,
            Err(status) => return status,
        };
        // SAFETY: `out` was checked non-null above.
        unsafe {
            *out = SlLanes {
                lambda1: pair.wavelength1(),
                lambda2: pair.wavelength2(),
                wide_lane: pair.wide_lane_wavelength(),
                narrow_lane: pair.narrow_lane_wavelength(),
            };
        }
        SL_OK
    })
}

/// The four two-frequency combinations differ only in which one they are, so they share one body.
///
/// # Safety
/// `out` is writable for one double.
unsafe fn dual_frequency(
    who: &str,
    f1: f64,
    f2: f64,
    combine: impl FnOnce(&FrequencyPair) -> f64,
    out: *mut f64,
) -> i32 {
    if out.is_null() {
        return fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: null out"));
    }
    match to_pair(who, f1, f2) {
        // SAFETY: `out` was checked non-null above.
        Ok(pair) => unsafe {
            *out = combine(&pair);
            SL_OK
        },
        Err(status) => status,
    }
}

/// Ionosphere-free combination `(f1²x1 - f2²x2)/(f1² - f2²)`, metres.
///
/// # Safety
/// `out` is writable for one double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ionosphere_free(
    f1: f64,
    f2: f64,
    x1: f64,
    x2: f64,
    out: *mut f64,
) -> i32 {
    // SAFETY: forwarding the caller's contract on `out`.
    guard(|| unsafe {
        dual_frequency(
            "sl_ionosphere_free",
            f1,
            f2,
            |p| p.ionosphere_free(x1, x2),
            out,
        )
    })
}

/// Wide-lane combination `(f1x1 - f2x2)/(f1 - f2)`, metres.
///
/// # Safety
/// `out` is writable for one double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_wide_lane(f1: f64, f2: f64, x1: f64, x2: f64, out: *mut f64) -> i32 {
    // SAFETY: forwarding the caller's contract on `out`.
    guard(|| unsafe { dual_frequency("sl_wide_lane", f1, f2, |p| p.wide_lane(x1, x2), out) })
}

/// Narrow-lane combination `(f1x1 + f2x2)/(f1 + f2)`, metres.
///
/// # Safety
/// `out` is writable for one double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_narrow_lane(f1: f64, f2: f64, x1: f64, x2: f64, out: *mut f64) -> i32 {
    // SAFETY: forwarding the caller's contract on `out`.
    guard(|| unsafe { dual_frequency("sl_narrow_lane", f1, f2, |p| p.narrow_lane(x1, x2), out) })
}

/// Melbourne-Wübbena combination: the wide-lane phase less the narrow-lane code.
///
/// In metres, or in wide-lane cycles with `cycles` set — the unit the arc is read in, since over a
/// clean arc the value hovers about an integer and a step in it is a wide-lane slip.
///
/// # Safety
/// `out` is writable for one double.
#[allow(clippy::too_many_arguments)] // two frequencies, two codes, two phases, the unit and the out
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_melbourne_wubbena(
    f1: f64,
    f2: f64,
    p1: f64,
    p2: f64,
    l1: f64,
    l2: f64,
    cycles: bool,
    out: *mut f64,
) -> i32 {
    // SAFETY: forwarding the caller's contract on `out`.
    guard(|| unsafe {
        dual_frequency(
            "sl_melbourne_wubbena",
            f1,
            f2,
            |pair| {
                if cycles {
                    pair.melbourne_wubbena_cycles(p1, p2, l1, l2)
                } else {
                    pair.melbourne_wubbena(p1, p2, l1, l2)
                }
            },
            out,
        )
    })
}

/// Geometry-free combination `x1 - x2`, metres — it takes no frequencies because everything
/// non-dispersive cancels. Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_geometry_free(x1: f64, x2: f64) -> f64 {
    guard_f64(|| gnss_geometry_free(x1, x2))
}

/// GRAPHIC combination `(p + l)/2`, metres — the single-frequency ionosphere-free observable.
/// Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_graphic(p: f64, l: f64) -> f64 {
    guard_f64(|| gnss_graphic(p, l))
}

fn slip_out(verdict: SlipVerdict) -> SlSlipVerdict {
    match verdict {
        SlipVerdict::ArcStart => SlSlipVerdict {
            verdict: SLIP_ARC_START,
            statistic: f64::NAN,
        },
        SlipVerdict::Continuous { statistic } => SlSlipVerdict {
            verdict: SLIP_CONTINUOUS,
            statistic,
        },
        SlipVerdict::Slipped { statistic } => SlSlipVerdict {
            verdict: SLIP_SLIPPED,
            statistic,
        },
    }
}

/// Both detectors take the same series and write the same verdicts, so they share one body.
///
/// # Safety
/// `times` and `values` are readable for `count` doubles; `out` is writable for `count`
/// `sl_slip_verdict`.
unsafe fn slip_scan(
    who: &str,
    times: *const f64,
    values: *const f64,
    count: usize,
    mut update: impl FnMut(f64, f64) -> SlipVerdict,
    out: *mut SlSlipVerdict,
) -> i32 {
    if times.is_null() || values.is_null() || out.is_null() {
        return fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: null pointer"));
    }
    if let Err(status) = check_nonzero(who, "count", count) {
        return status;
    }
    // SAFETY: checked non-null; the caller states both are readable for `count`.
    let (t, v) = unsafe {
        (
            core::slice::from_raw_parts(times, count),
            core::slice::from_raw_parts(values, count),
        )
    };
    for (i, (&ti, &vi)) in t.iter().zip(v).enumerate() {
        // SAFETY: `out` is writable for `count`, and `i` is below it.
        unsafe { *out.add(i) = slip_out(update(ti, vi)) };
    }
    SL_OK
}

/// Run the first-difference cycle-slip detector over a series, one verdict per sample.
///
/// # Safety
/// `times` and `values` are readable for `count` doubles; `out` is writable for `count`
/// `sl_slip_verdict`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_detect_cycle_slips_geometry_free(
    times: *const f64,
    values: *const f64,
    count: usize,
    threshold: f64,
    max_gap: f64,
    out: *mut SlSlipVerdict,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_detect_cycle_slips_geometry_free";
        let mut detector = match GeometryFreeDetector::new(threshold, max_gap) {
            Ok(detector) => detector,
            Err(e) => return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: {e}")),
        };
        // SAFETY: forwarding the caller's contract on the three pointers.
        unsafe { slip_scan(WHO, times, values, count, |t, v| detector.update(t, v), out) }
    })
}

/// Run the polynomial-fit cycle-slip detector over a series, one verdict per sample.
///
/// The first `window` samples are `SL_SLIP_ARC_START`: nothing is tested until the window fills.
///
/// # Safety
/// As `sl_detect_cycle_slips_geometry_free`.
#[allow(clippy::too_many_arguments)] // the series and its length, the fit's four settings, the out
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_detect_cycle_slips_polynomial(
    times: *const f64,
    values: *const f64,
    count: usize,
    order: usize,
    window: usize,
    threshold: f64,
    max_gap: f64,
    out: *mut SlSlipVerdict,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_detect_cycle_slips_polynomial";
        let mut detector = match PolynomialFitDetector::new(order, window, threshold, max_gap) {
            Ok(detector) => detector,
            Err(e) => return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: {e}")),
        };
        // SAFETY: forwarding the caller's contract on the three pointers.
        unsafe { slip_scan(WHO, times, values, count, |t, v| detector.update(t, v), out) }
    })
}

/// The decorrelated problem all three ambiguity entry points start from.
///
/// # Safety
/// `float_ambiguities` is readable for `n` doubles and `covariance` for `n * n`.
unsafe fn read_reduction(
    who: &str,
    float_ambiguities: *const f64,
    n: usize,
    covariance: *const f64,
) -> Result<Reduction, i32> {
    if float_ambiguities.is_null() || covariance.is_null() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: null pointer"),
        ));
    }
    check_nonzero(who, "n", n)?;
    let Some(entries) = n.checked_mul(n) else {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: n is {n}, whose square does not fit a size_t"),
        ));
    };
    // SAFETY: checked non-null; the caller states both extents.
    let (float, q) = unsafe {
        (
            core::slice::from_raw_parts(float_ambiguities, n),
            core::slice::from_raw_parts(covariance, entries),
        )
    };
    Reduction::new(float, q).map_err(|e| fail(SL_ERR_INVALID_ARGUMENT, format!("{who}: {e}")))
}

/// Decorrelate an ambiguity problem: the integer `Z` of LAMBDA, row-major, and the conditional
/// variances whose product is the volume of the search ellipsoid.
///
/// # Safety
/// `float_ambiguities` is readable for `n` doubles, `covariance` for `n * n`;
/// `out_transformation` is writable for `n * n` doubles and `out_conditional` for `n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ambiguity_decorrelation(
    float_ambiguities: *const f64,
    n: usize,
    covariance: *const f64,
    out_transformation: *mut f64,
    out_conditional: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_ambiguity_decorrelation";
        if out_transformation.is_null() || out_conditional.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out"));
        }
        // SAFETY: forwarding the caller's contract on the two input pointers.
        let reduced = match unsafe { read_reduction(WHO, float_ambiguities, n, covariance) } {
            Ok(reduced) => reduced,
            Err(status) => return status,
        };
        let z = reduced.transformation();
        let d = reduced.conditional_variances();
        // SAFETY: `out_transformation` is writable for `n * n` and `z` is exactly that long;
        // `out_conditional` for `n`, and `d` is exactly that long.
        unsafe {
            core::ptr::copy_nonoverlapping(z.as_ptr(), out_transformation, z.len());
            core::ptr::copy_nonoverlapping(d.as_ptr(), out_conditional, d.len());
        }
        SL_OK
    })
}

/// The bootstrapped integer solution and its objective — one pass of sequential conditional
/// rounding, which bounds the search but is not in general the integer least-squares solution.
///
/// # Safety
/// As `sl_ambiguity_decorrelation`, with `out_ambiguities` writable for `n` `int64_t` and
/// `out_squared_norm` for one double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_ambiguity_bootstrap(
    float_ambiguities: *const f64,
    n: usize,
    covariance: *const f64,
    out_ambiguities: *mut i64,
    out_squared_norm: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_ambiguity_bootstrap";
        if out_ambiguities.is_null() || out_squared_norm.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out"));
        }
        // SAFETY: forwarding the caller's contract on the two input pointers.
        let reduced = match unsafe { read_reduction(WHO, float_ambiguities, n, covariance) } {
            Ok(reduced) => reduced,
            Err(status) => return status,
        };
        let candidate = reduced.bootstrap();
        // SAFETY: `out_ambiguities` is writable for `n`, and the candidate is exactly `n` long.
        unsafe {
            core::ptr::copy_nonoverlapping(
                candidate.ambiguities.as_ptr(),
                out_ambiguities,
                candidate.ambiguities.len(),
            );
            *out_squared_norm = candidate.squared_norm;
        }
        SL_OK
    })
}

/// The best integer ambiguity vectors by the LAMBDA search, best first, in the caller's own
/// parametrisation. The ratio of the second squared norm to the first is the acceptance test.
///
/// Writes `*out_len` candidates, which is at most `count`; the search can hold fewer.
///
/// # Safety
/// As `sl_ambiguity_decorrelation`, with `out_ambiguities` writable for `count * n` `int64_t`,
/// `out_squared_norms` for `count` doubles, and `out_len` for one `size_t`.
#[allow(clippy::too_many_arguments)] // the problem's three inputs, the count, and three outputs
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_resolve_ambiguities(
    float_ambiguities: *const f64,
    n: usize,
    covariance: *const f64,
    count: usize,
    out_ambiguities: *mut i64,
    out_squared_norms: *mut f64,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_resolve_ambiguities";
        if out_ambiguities.is_null() || out_squared_norms.is_null() || out_len.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null out"));
        }
        if let Err(status) = check_nonzero(WHO, "count", count) {
            return status;
        }
        // SAFETY: forwarding the caller's contract on the two input pointers.
        let reduced = match unsafe { read_reduction(WHO, float_ambiguities, n, covariance) } {
            Ok(reduced) => reduced,
            Err(status) => return status,
        };
        let found = match reduced.search(count) {
            Ok(found) => found,
            Err(e) => return fail(SL_ERR_COMPUTE, format!("{WHO}: {e}")),
        };
        // SAFETY: `out_len` was checked non-null above.
        unsafe { *out_len = found.len() };
        for (i, candidate) in found.iter().enumerate() {
            // SAFETY: `found.len() <= count`, and each candidate is `n` long, so `i * n + n` stays
            // inside the `count * n` the caller states is writable.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    candidate.ambiguities.as_ptr(),
                    out_ambiguities.add(i * n),
                    candidate.ambiguities.len(),
                );
                *out_squared_norms.add(i) = candidate.squared_norm;
            }
        }
        SL_OK
    })
}

// ── Conjunction assessment ────────────────────────────────────────────────────────────────────
//
// #92's collision probability, #315's screen and #322's domain check. The two scans take a callback
// for the relative state, on the precedent `sl_gauss_newton` set: a caller that cannot produce a
// sample says so through the status, and the non-finite sample this side substitutes is what stops
// the Rust scan at the epoch the data stopped.

/// `sl_conjunction`: the relative geometry at TCA and the two hard-body radii.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlConjunction {
    /// `r2 - r1` at TCA, metres, in one common inertial frame.
    pub relative_position: SlVec3,
    /// `v2 - v1` at TCA, m/s, in that frame.
    pub relative_velocity: SlVec3,
    /// The primary's 3x3 position covariance, row-major, m².
    pub primary_covariance: [f64; 9],
    /// The secondary's, in the same frame.
    pub secondary_covariance: [f64; 9],
    /// The primary's equivalent-sphere radius, metres.
    pub primary_radius: f64,
    /// The secondary's, metres.
    pub secondary_radius: f64,
}

/// `sl_collision_probability`: the 2-D probability by all four estimators.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlCollisionProbability {
    /// Foster's numerical integration of the encounter-plane Gaussian.
    pub foster: f64,
    /// Patera's contour integral of the same.
    pub patera: f64,
    /// Chan's equivalent-area series.
    pub chan: f64,
    /// The maximum over a scaling of the covariance — a bound, not an estimate.
    pub maximum: f64,
}

/// `sl_relative_sample`: a relative state and the joint covariance `C1 + C2` at one epoch.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlRelativeSample {
    /// Relative position, metres.
    pub position: SlVec3,
    /// Relative velocity, m/s.
    pub velocity: SlVec3,
    /// The joint 3x3 position covariance, row-major, m².
    pub covariance: [f64; 9],
}

/// `sl_usage_violations`: whether the 2-D probability is inside its own model's domain.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlUsageViolations {
    /// Whether any of the four checks fired.
    pub any: bool,
    /// A sampled covariance was not positive definite.
    pub non_positive_definite: bool,
    /// The rectilinear and curvilinear encounters put the minimum at different offsets.
    pub offset: bool,
    /// The encounter is extended in time relative to the shorter orbital period.
    pub extended: bool,
    /// The two encounters disagree enough that the 2-D probability is not known to be accurate.
    pub inaccuracy: bool,
    /// The offset check's dimensionless statistic.
    pub offset_indicator: f64,
    /// The extended-encounter statistic.
    pub extended_indicator: f64,
    /// The inaccuracy statistic.
    pub inaccuracy_indicator: f64,
    /// The rough multiplicative correction: `Pc ~ correction_factor * Pc_2D`.
    pub correction_factor: f64,
}

/// `sl_close_approach`: one local minimum of the separation.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlCloseApproach {
    /// Epoch of closest approach, seconds, in the scan's own timescale.
    pub tca: f64,
    /// Separation there, metres.
    pub distance: f64,
    /// Relative speed there, m/s.
    pub relative_speed: f64,
}

/// The callback `sl_closest_approaches` calls: fill `out` with the relative state at `t`, return
/// `SL_OK`. Anything else ends the scan.
pub type SlRelativeStateFn =
    Option<unsafe extern "C" fn(user_data: *mut c_void, t: f64, out: *mut SlState) -> i32>;

/// The callback `sl_conjunction_usage_violations` calls, which additionally carries the covariance.
pub type SlRelativeSampleFn =
    Option<unsafe extern "C" fn(user_data: *mut c_void, t: f64, out: *mut SlRelativeSample) -> i32>;

/// The 3x3 covariance as the crate's nested form.
fn to_three_by_three(flat: [f64; 9]) -> [[f64; 3]; 3] {
    [
        [flat[0], flat[1], flat[2]],
        [flat[3], flat[4], flat[5]],
        [flat[6], flat[7], flat[8]],
    ]
}

/// The geometry both conjunction entry points start from.
fn to_conjunction(c: SlConjunction) -> ConjunctionGeometry {
    ConjunctionGeometry::new(
        to_three(c.relative_position),
        to_three(c.relative_velocity),
        to_three_by_three(c.primary_covariance),
        to_three_by_three(c.secondary_covariance),
        c.primary_radius,
        c.secondary_radius,
    )
}

/// The 2-D collision probability at one conjunction, by all four estimators.
///
/// # Safety
/// `geometry` points at one initialised `sl_conjunction`; `out` is writable for one
/// `sl_collision_probability`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_collision_probability(
    geometry: *const SlConjunction,
    out: *mut SlCollisionProbability,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_collision_probability";
        if geometry.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let plane = match to_conjunction(unsafe { *geometry }).project() {
            Ok(plane) => plane,
            Err(e) => return fail(SL_ERR_DOMAIN, format!("{WHO}: {e}")),
        };
        // SAFETY: `out` was checked non-null above.
        unsafe {
            *out = SlCollisionProbability {
                foster: plane.probability_foster(),
                patera: plane.probability_patera(),
                chan: plane.probability_chan(),
                maximum: plane.probability_max(),
            };
        }
        SL_OK
    })
}

/// Whether the 2-D collision probability is inside its own model's domain.
///
/// `sample(t)` supplies the relative state and the joint covariance `t` seconds from TCA, under
/// whatever dynamics the caller propagates with. `minimum_period` is the shorter of the two objects'
/// orbital periods, seconds — the scale the published thresholds are stated on.
///
/// # Safety
/// `geometry` points at one initialised `sl_conjunction`; `out` is writable for one
/// `sl_usage_violations`. `sample` is called with `user_data` unchanged and must not unwind.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_conjunction_usage_violations(
    geometry: *const SlConjunction,
    minimum_period: f64,
    sample: SlRelativeSampleFn,
    user_data: *mut c_void,
    out: *mut SlUsageViolations,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_conjunction_usage_violations";
        if geometry.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let Some(call) = sample else {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null sample"));
        };
        // SAFETY: checked non-null; the caller states it is initialised.
        let assessed = to_conjunction(unsafe { *geometry });
        // The first refusal is held rather than propagated: the Rust check takes a closure that
        // returns a value, not a `Result`, so the way to stop it is the non-finite sample it
        // already treats as the end of the data. The status is what the caller is told, so a
        // sampler that failed is never reported as a check that simply found nothing wrong.
        let status = Cell::new(SL_OK);
        let unusable = RelativeSample {
            position: [f64::NAN; 3],
            velocity: [f64::NAN; 3],
            covariance: [[f64::NAN; 3]; 3],
        };
        let violations = estimation_usage_violations(&assessed, minimum_period, |t| {
            if status.get() != SL_OK {
                return unusable;
            }
            let mut slot = SlRelativeSample {
                position: SlVec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                velocity: SlVec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                covariance: [0.0; 9],
            };
            // SAFETY: the caller states `call` is a valid sampler; `slot` is a live local.
            let code = unsafe { call(user_data, t, &raw mut slot) };
            if code != SL_OK {
                status.set(fail(
                    SL_ERR_COMPUTE,
                    format!("{WHO}: the sampler refused at t = {t}, returning {code}"),
                ));
                return unusable;
            }
            RelativeSample {
                position: to_three(slot.position),
                velocity: to_three(slot.velocity),
                covariance: to_three_by_three(slot.covariance),
            }
        });
        // The sampler's refusal outranks whatever the check made of the samples it never got.
        if status.get() != SL_OK {
            return status.get();
        }
        let v = match violations {
            Ok(v) => v,
            Err(e) => return fail(SL_ERR_DOMAIN, format!("{WHO}: {e}")),
        };
        // SAFETY: `out` was checked non-null above.
        unsafe {
            *out = SlUsageViolations {
                any: v.any(),
                non_positive_definite: v.non_positive_definite(),
                offset: v.offset(),
                extended: v.extended(),
                inaccuracy: v.inaccuracy(),
                offset_indicator: v.offset_indicator(),
                extended_indicator: v.extended_indicator(),
                inaccuracy_indicator: v.inaccuracy_indicator(),
                correction_factor: v.correction_factor(),
            };
        }
        SL_OK
    })
}

/// Every closest approach of a relative trajectory in `[start, end]`, in time order. Two-call
/// protocol; note that sizing runs the scan, so the callback is invoked by both calls.
///
/// # Safety
/// `out` is null or writable for `cap` `sl_close_approach`; `out_len` is writable for one `size_t`.
/// `relative` is called with `user_data` unchanged and must not unwind.
#[allow(clippy::too_many_arguments)] // the sampler and its data, the span, the two tolerances, out
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_closest_approaches(
    relative: SlRelativeStateFn,
    user_data: *mut c_void,
    start: f64,
    end: f64,
    step: f64,
    tolerance: f64,
    out: *mut SlCloseApproach,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_closest_approaches";
        let Some(call) = relative else {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null relative"));
        };
        // As `sl_conjunction_usage_violations`: the refusal is held, and the non-finite sample is
        // what ends the Rust scan at the epoch the caller's data stopped.
        let status = Cell::new(SL_OK);
        let unusable = ([f64::NAN; 3], [f64::NAN; 3]);
        let mut found: Vec<SlCloseApproach> = Vec::new();
        let scanned = estimation_closest_approaches(
            |t| {
                if status.get() != SL_OK {
                    return unusable;
                }
                let mut slot = SlState {
                    position: SlVec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    velocity: SlVec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                };
                // SAFETY: the caller states `call` is a valid sampler; `slot` is a live local.
                let code = unsafe { call(user_data, t, &raw mut slot) };
                if code != SL_OK {
                    status.set(fail(
                        SL_ERR_COMPUTE,
                        format!("{WHO}: the sampler refused at t = {t}, returning {code}"),
                    ));
                    return unusable;
                }
                (to_three(slot.position), to_three(slot.velocity))
            },
            start,
            end,
            step,
            tolerance,
            &mut |approach| {
                found.push(SlCloseApproach {
                    tca: approach.tca,
                    distance: approach.distance,
                    relative_speed: approach.relative_speed,
                });
            },
        );
        if status.get() != SL_OK {
            return status.get();
        }
        if let Err(e) = scanned {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: {e}"));
        }
        // SAFETY: forwarding the caller's own contract on `out`, `cap` and `out_len`.
        unsafe { deliver(&found, out, cap, out_len) }
    })
}

/// Whether two orbits are far enough apart in radius that no conjunction is possible — the first
/// and cheapest stage of an all-on-all screen. `true` is a geometric certainty; the converse is not
/// claimed. A non-finite input answers `false`, so a corrupt element set is examined rather than
/// silently discarded. Total.
#[unsafe(no_mangle)]
pub extern "C" fn sl_radially_separated(
    perigee_a: f64,
    apogee_a: f64,
    perigee_b: f64,
    apogee_b: f64,
    threshold: f64,
) -> bool {
    // No `guard`: `guard` reports a contained panic through a status this function does not have,
    // and the predicate below is arithmetic and comparisons on five doubles, which cannot panic.
    estimation_radially_separated(perigee_a, apogee_a, perigee_b, apogee_b, threshold)
}

// ── Closed-loop control ───────────────────────────────────────────────────────────────────────

/// `sl_latency`: the three delays of a feedback loop, seconds.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlLatency {
    /// The measurement describes the state this long ago.
    pub sensor_s: f64,
    /// Deciding takes this long.
    pub compute_s: f64,
    /// The actuator takes this long to respond.
    pub actuator_s: f64,
}

/// `sl_attitude_state`: an orientation and a body rate.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlAttitudeState {
    /// Inertial→body change of basis, row-major.
    pub rotation: [f64; 9],
    /// Body rate in body axes, rad/s.
    pub rate: SlVec3,
}

/// `sl_loop_sample`: one step of a closed-loop run.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlLoopSample {
    /// Seconds since the run started.
    pub time_s: f64,
    /// The principal angle of the error the controller acted on.
    pub error_angle_rad: f64,
    /// The moment commanded at this step, N·m, body axes.
    pub torque: SlVec3,
}

/// `sl_delay_margin`: the delay a linearised loop tolerates, and the loop it was measured on.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlDelayMargin {
    /// The largest feedback delay the loop survives, seconds.
    pub seconds: f64,
    /// The frequency the pole pair crosses at, rad/s.
    pub crossing_frequency_rad_s: f64,
    /// The loop's undamped natural frequency, rad/s.
    pub natural_frequency_rad_s: f64,
    /// The loop's damping ratio.
    pub damping_ratio: f64,
    /// `τ_max · ω_n`, which depends only on the damping ratio.
    pub normalised: f64,
}

/// `sl_element_feedback`: the station-keeping law's weights, gain and acceleration limit.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlElementFeedback {
    /// One Bryson weight per equinoctial element.
    pub weights: [f64; 6],
    /// The proportional gain.
    pub proportional: f64,
    /// The largest acceleration the thruster can produce, m/s².
    pub max_acceleration: f64,
}

/// `sl_attitude_law`: Wie's quaternion PD.
const LAW_QUATERNION_PD: i32 = 0;
/// `sl_attitude_law`: Schaub & Junkins' MRP Lyapunov law.
const LAW_MRP_LYAPUNOV: i32 = 1;

/// `sl_consensus_protocol`: the relative measurement itself is delayed.
const CONSENSUS_DELAYED_RELATIVE: i32 = 0;
/// `sl_consensus_protocol`: only the neighbour's state is late.
const CONSENSUS_CURRENT_SELF: i32 = 1;

/// The history capacity this ABI gives an attitude loop, in samples.
///
/// `History<12, C>` stores its ring inline, so this is `C · 12 · 8` bytes of stack — 49 KB. The
/// Rust API makes the capacity a const parameter so a caller can size it; a C caller has no const
/// parameter to give, so the ABI picks one and refuses a lag that does not fit rather than
/// wrapping the ring and reading a state from a previous era.
const CAPI_LOOP_HISTORY: usize = 512;

/// The smallest and largest agent count the consensus entry points are instantiated for.
const MIN_AGENTS: usize = 2;
/// See [`MIN_AGENTS`].
const MAX_AGENTS: usize = 12;

/// Either attitude law, behind one type, because every control entry point takes the same
/// `(law, proportional, derivative)` triple and the two are interchangeable exactly where the
/// crate makes them interchangeable.
#[derive(Clone, Copy)]
enum CapiLaw {
    QuaternionPd(QuaternionPd),
    MrpLyapunov(MrpLyapunov),
}

impl ControlLaw for CapiLaw {
    fn torque(&self, error: &AttitudeError, inertia: &Inertia) -> [f64; 3] {
        match self {
            Self::QuaternionPd(law) => law.torque(error, inertia),
            Self::MrpLyapunov(law) => law.torque(error, inertia),
        }
    }
}

impl SmallAngleGains for CapiLaw {
    fn small_angle_gains(&self) -> (f64, f64) {
        match self {
            Self::QuaternionPd(law) => law.small_angle_gains(),
            Self::MrpLyapunov(law) => law.small_angle_gains(),
        }
    }
}

/// The law an `sl_attitude_law` tag and a gain pair name, or the refusal that says which failed.
fn check_law(who: &str, law: i32, proportional: f64, derivative: f64) -> Result<CapiLaw, i32> {
    let built = match law {
        LAW_QUATERNION_PD => QuaternionPd::new(proportional, derivative).map(CapiLaw::QuaternionPd),
        LAW_MRP_LYAPUNOV => MrpLyapunov::new(proportional, derivative).map(CapiLaw::MrpLyapunov),
        other => {
            return Err(fail(
                SL_ERR_INVALID_ARGUMENT,
                format!("{who}: law is {other}, and must be 0 or 1"),
            ));
        }
    };
    built.ok_or_else(|| {
        fail(
            SL_ERR_DOMAIN,
            format!(
                "{who}: the gains ({proportional}, {derivative}) must both be finite and positive"
            ),
        )
    })
}

/// The inertia tensor nine row-major entries describe, or the refusal.
fn check_inertia(who: &str, flat: [f64; 9]) -> Result<Inertia, i32> {
    Inertia::from_tensor(to_three_by_three(flat)).ok_or_else(|| {
        fail(
            SL_ERR_DOMAIN,
            format!("{who}: the inertia tensor must be symmetric and positive definite"),
        )
    })
}

/// The latency three numbers describe, or the refusal.
fn check_latency(who: &str, latency: SlLatency) -> Result<Latency, i32> {
    Latency::new(latency.sensor_s, latency.compute_s, latency.actuator_s).ok_or_else(|| {
        fail(
            SL_ERR_INVALID_ARGUMENT,
            format!(
                "{who}: each latency must be finite and non-negative, not ({}, {}, {})",
                latency.sensor_s, latency.compute_s, latency.actuator_s
            ),
        )
    })
}

/// The station-keeping law a `sl_element_feedback` describes, or the refusal.
fn check_element_feedback(who: &str, law: SlElementFeedback) -> Result<ElementFeedback, i32> {
    ElementFeedback::new(law.weights, law.proportional, law.max_acceleration).ok_or_else(|| {
        fail(
            SL_ERR_DOMAIN,
            format!(
                "{who}: every weight must be finite and non-negative with one positive, and the \
                 gain and the acceleration limit finite and positive"
            ),
        )
    })
}

/// The protocol an `sl_consensus_protocol` tag names.
fn check_protocol(who: &str, protocol: i32) -> Result<Protocol, i32> {
    match protocol {
        CONSENSUS_DELAYED_RELATIVE => Ok(Protocol::DelayedRelative),
        CONSENSUS_CURRENT_SELF => Ok(Protocol::CurrentSelf),
        other => Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: protocol is {other}, and must be 0 or 1"),
        )),
    }
}

/// An agent count this ABI has an instantiation for.
fn check_agents(who: &str, agents: usize) -> Result<(), i32> {
    if !(MIN_AGENTS..=MAX_AGENTS).contains(&agents) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!(
                "{who}: {agents} agents, and this ABI is built for {MIN_AGENTS} to {MAX_AGENTS}"
            ),
        ));
    }
    Ok(())
}

/// A matrix order this ABI has a `characteristic_polynomial` instantiation for.
///
/// The same two numbers as `check_agents` and not the same wall: this one is the order of a
/// caller's matrix, that one the size of a formation, and a refusal that named the wrong noun
/// would send the reader to the wrong fix.
fn check_matrix_order(who: &str, n: usize) -> Result<(), i32> {
    if !(MIN_AGENTS..=MAX_AGENTS).contains(&n) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: order {n}, and this ABI is built for {MIN_AGENTS} to {MAX_AGENTS}"),
        ));
    }
    Ok(())
}

/// The C triple a Rust one becomes on the way out. The inverse of `to_three`.
fn from_three(v: [f64; 3]) -> SlVec3 {
    SlVec3 {
        x: v[0],
        y: v[1],
        z: v[2],
    }
}

/// A `ROWS × COLS` matrix out of a flat row-major slice the caller has already been checked to
/// have supplied enough of.
fn to_rows<const COLS: usize, const ROWS: usize>(flat: &[f64]) -> [[f64; COLS]; ROWS] {
    core::array::from_fn(|i| core::array::from_fn(|j| flat[i * COLS + j]))
}

/// The margin of a linearised axis, in the shape the header publishes it.
fn to_delay_margin(axis: &SecondOrderLoop) -> SlDelayMargin {
    let margin = axis.delay_margin();
    SlDelayMargin {
        seconds: margin.seconds(),
        crossing_frequency_rad_s: margin.crossing_frequency(),
        natural_frequency_rad_s: axis.natural_frequency(),
        damping_ratio: axis.damping_ratio(),
        normalised: axis.normalised_delay_margin(),
    }
}

/// The graph a flat adjacency matrix describes, or the refusal.
fn to_graph<const N: usize>(who: &str, adjacency: &[f64]) -> Result<Graph<N>, i32> {
    Graph::new(to_rows::<N, N>(adjacency)).map_err(|e| {
        fail(
            SL_ERR_DOMAIN,
            format!("{who}: not a usable interaction graph: {e:?}"),
        )
    })
}

/// The total latency `τ_s + τ_c + τ_a`. See the header.
///
/// # Safety
/// `latency` points at one initialised `sl_latency`; `out` is writable for one `double`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_latency_total(latency: *const SlLatency, out: *mut f64) -> i32 {
    guard(|| {
        const WHO: &str = "sl_latency_total";
        if latency.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let latency = match check_latency(WHO, unsafe { *latency }) {
            Ok(l) => l,
            Err(status) => return status,
        };
        // SAFETY: checked non-null.
        unsafe { *out = latency.total() };
        SL_OK
    })
}

/// How many whole steps of `step` the total latency is. See the header.
///
/// # Safety
/// `latency` points at one initialised `sl_latency`; `out` is writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_latency_steps(
    latency: *const SlLatency,
    step: f64,
    out: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_latency_steps";
        if latency.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_positive(WHO, "step", step, SL_ERR_INVALID_ARGUMENT) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let latency = match check_latency(WHO, unsafe { *latency }) {
            Ok(l) => l,
            Err(status) => return status,
        };
        let Some(steps) = latency.steps(step) else {
            return fail(
                SL_ERR_DOMAIN,
                format!(
                    "{WHO}: a latency of {} s is not a whole number of {step} s steps",
                    latency.total()
                ),
            );
        };
        // SAFETY: checked non-null.
        unsafe { *out = steps };
        SL_OK
    })
}

/// The attitude error between two frames. See the header.
///
/// # Safety
/// `rotation`, `reference` and `inertia` are readable for nine `double`; `rate` and
/// `rate_reference` point at initialised `sl_vec3`; the out-pointers are writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_attitude_error_mrp(
    rotation: *const f64,
    reference: *const f64,
    rate: *const SlVec3,
    rate_reference: *const SlVec3,
    out_mrp: *mut SlVec3,
    out_angle_rad: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_attitude_error_mrp";
        if out_mrp.is_null() || out_angle_rad.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: the caller states the four inputs are readable at the documented sizes.
        let frames = unsafe { read_attitude_pair(WHO, rotation, reference, rate, rate_reference) };
        let error = match frames {
            Ok(error) => error,
            Err(status) => return status,
        };
        // SAFETY: both checked non-null above.
        unsafe {
            *out_mrp = from_three(error.mrp());
            *out_angle_rad = error.angle();
        }
        SL_OK
    })
}

/// The four pointers every attitude entry point starts from, read and validated once.
///
/// # Safety
/// As the entry points that call it: two nine-`double` matrices and two `sl_vec3`.
unsafe fn read_attitude_pair(
    who: &str,
    rotation: *const f64,
    reference: *const f64,
    rate: *const SlVec3,
    rate_reference: *const SlVec3,
) -> Result<AttitudeError, i32> {
    if rotation.is_null() || reference.is_null() || rate.is_null() || rate_reference.is_null() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: null pointer"),
        ));
    }
    // SAFETY: checked non-null; the caller states the sizes.
    let (rotation, reference, rate, rate_reference) = unsafe {
        (
            core::slice::from_raw_parts(rotation, 9),
            core::slice::from_raw_parts(reference, 9),
            *rate,
            *rate_reference,
        )
    };
    check_all_finite(who, "rotation", rotation)?;
    check_all_finite(who, "reference", reference)?;
    Ok(AttitudeError::new(
        &to_rows::<3, 3>(rotation),
        &to_rows::<3, 3>(reference),
        to_three(rate),
        to_three(rate_reference),
    ))
}

/// The commanded body moment. See the header.
///
/// # Safety
/// As `sl_attitude_error_mrp`, plus `inertia` readable for nine `double` and `out` writable for one
/// `sl_vec3`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn sl_attitude_torque(
    law: i32,
    proportional: f64,
    derivative: f64,
    rotation: *const f64,
    reference: *const f64,
    rate: *const SlVec3,
    rate_reference: *const SlVec3,
    inertia: *const f64,
    out: *mut SlVec3,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_attitude_torque";
        if inertia.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let law = match check_law(WHO, law, proportional, derivative) {
            Ok(law) => law,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states it is readable for nine.
        let tensor = unsafe { read_nine(WHO, inertia, "inertia") };
        let inertia = match tensor.and_then(|flat| check_inertia(WHO, flat)) {
            Ok(inertia) => inertia,
            Err(status) => return status,
        };
        // SAFETY: the caller states the four inputs are readable at the documented sizes.
        let error =
            match unsafe { read_attitude_pair(WHO, rotation, reference, rate, rate_reference) } {
                Ok(error) => error,
                Err(status) => return status,
            };
        // SAFETY: checked non-null above.
        unsafe { *out = from_three(law.torque(&error, &inertia)) };
        SL_OK
    })
}

/// Nine finite `double` from a caller's pointer.
///
/// # Safety
/// `ptr` is null or readable for nine `double`.
unsafe fn read_nine(who: &str, ptr: *const f64, name: &str) -> Result<[f64; 9], i32> {
    if ptr.is_null() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: null {name} pointer"),
        ));
    }
    // SAFETY: checked non-null; the caller states it is readable for nine.
    let flat = unsafe { core::slice::from_raw_parts(ptr, 9) };
    check_all_finite(who, name, flat)?;
    Ok(core::array::from_fn(|i| flat[i]))
}

/// Six `double` from a caller's pointer, finiteness left to the caller's own predicate.
///
/// # Safety
/// `ptr` is null or readable for six `double`.
unsafe fn read_six(who: &str, ptr: *const f64, name: &str) -> Result<[f64; 6], i32> {
    if ptr.is_null() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: null {name} pointer"),
        ));
    }
    // SAFETY: checked non-null; the caller states it is readable for six.
    let flat = unsafe { core::slice::from_raw_parts(ptr, 6) };
    Ok(core::array::from_fn(|i| flat[i]))
}

/// The MRP law's Lyapunov function. See the header.
///
/// # Safety
/// As `sl_attitude_torque`, with `out` writable for one `double`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn sl_attitude_lyapunov(
    proportional: f64,
    derivative: f64,
    rotation: *const f64,
    reference: *const f64,
    rate: *const SlVec3,
    rate_reference: *const SlVec3,
    inertia: *const f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_attitude_lyapunov";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let law = match check_law(WHO, LAW_MRP_LYAPUNOV, proportional, derivative) {
            Ok(CapiLaw::MrpLyapunov(law)) => law,
            Ok(_) => unreachable!("the tag is the MRP one"),
            Err(status) => return status,
        };
        // SAFETY: checked non-null inside; the caller states it is readable for nine.
        let tensor = unsafe { read_nine(WHO, inertia, "inertia") };
        let inertia = match tensor.and_then(|flat| check_inertia(WHO, flat)) {
            Ok(inertia) => inertia,
            Err(status) => return status,
        };
        // SAFETY: the caller states the four inputs are readable at the documented sizes.
        let error =
            match unsafe { read_attitude_pair(WHO, rotation, reference, rate, rate_reference) } {
                Ok(error) => error,
                Err(status) => return status,
            };
        // SAFETY: checked non-null above.
        unsafe { *out = law.lyapunov(&error, &inertia) };
        SL_OK
    })
}

/// The law's small-angle linearisation. See the header.
///
/// # Safety
/// Both out-pointers are writable for one `double`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_attitude_small_angle_gains(
    law: i32,
    proportional: f64,
    derivative: f64,
    out_angle_gain: *mut f64,
    out_rate_gain: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_attitude_small_angle_gains";
        if out_angle_gain.is_null() || out_rate_gain.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let law = match check_law(WHO, law, proportional, derivative) {
            Ok(law) => law,
            Err(status) => return status,
        };
        let (angle, rate) = law.small_angle_gains();
        // SAFETY: both checked non-null above.
        unsafe {
            *out_angle_gain = angle;
            *out_rate_gain = rate;
        }
        SL_OK
    })
}

/// The critically-damped derivative gain. See the header.
///
/// # Safety
/// `out` is writable for one `double`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_quaternion_pd_critically_damped(
    proportional: f64,
    moment: f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_quaternion_pd_critically_damped";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let Some(law) = QuaternionPd::critically_damped(proportional, moment) else {
            return fail(
                SL_ERR_DOMAIN,
                format!(
                    "{WHO}: the gain {proportional} and the moment {moment} must both be finite \
                     and positive"
                ),
            );
        };
        // `P = ζ·K/ω_n`, since `ω_n = √(K/2I)` makes `K/ω_n = √(2IK)`. Recovered from the law's
        // own accessors rather than by writing `√(2 I K)` a second time here, where it could
        // drift from the crate's.
        // SAFETY: checked non-null above.
        unsafe { *out = law.damping_ratio(moment) * proportional / law.natural_frequency(moment) };
        SL_OK
    })
}

/// The quaternion PD law's second-order response. See the header.
///
/// # Safety
/// Both out-pointers are writable for one `double`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_quaternion_pd_response(
    proportional: f64,
    derivative: f64,
    moment: f64,
    out_natural_frequency_rad_s: *mut f64,
    out_damping_ratio: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_quaternion_pd_response";
        if out_natural_frequency_rad_s.is_null() || out_damping_ratio.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let law = match check_law(WHO, LAW_QUATERNION_PD, proportional, derivative) {
            Ok(CapiLaw::QuaternionPd(law)) => law,
            Ok(_) => unreachable!("the tag is the quaternion one"),
            Err(status) => return status,
        };
        if let Err(status) = check_positive(WHO, "moment", moment, SL_ERR_DOMAIN) {
            return status;
        }
        // SAFETY: both checked non-null above.
        unsafe {
            *out_natural_frequency_rad_s = law.natural_frequency(moment);
            *out_damping_ratio = law.damping_ratio(moment);
        }
        SL_OK
    })
}

/// Run an attitude loop at a fixed rate under a fixed latency. See the header.
///
/// # Safety
/// `latency`, `initial` and `seed` point at initialised structs; `reference` and `inertia` are
/// readable for nine `double`; `out` is null or writable for `cap` samples; `out_len` is writable;
/// `out_final` is null or writable for one `sl_attitude_state`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn sl_attitude_loop_run(
    law: i32,
    proportional: f64,
    derivative: f64,
    latency: *const SlLatency,
    step: f64,
    initial: *const SlAttitudeState,
    seed: *const SlAttitudeState,
    reference: *const f64,
    inertia: *const f64,
    steps: usize,
    out: *mut SlLoopSample,
    cap: usize,
    out_len: *mut usize,
    out_final: *mut SlAttitudeState,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_attitude_loop_run";
        if latency.is_null() || initial.is_null() || seed.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let law = match check_law(WHO, law, proportional, derivative) {
            Ok(law) => law,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states they are initialised.
        let (latency, initial, seed) = unsafe { (*latency, *initial, *seed) };
        let latency = match check_latency(WHO, latency) {
            Ok(latency) => latency,
            Err(status) => return status,
        };
        // SAFETY: checked inside; the caller states both are readable for nine.
        let matrices = unsafe {
            read_nine(WHO, reference, "reference").and_then(|reference| {
                read_nine(WHO, inertia, "inertia").map(|inertia| (reference, inertia))
            })
        };
        let (reference, inertia) = match matrices {
            Ok(pair) => pair,
            Err(status) => return status,
        };
        let inertia = match check_inertia(WHO, inertia) {
            Ok(inertia) => inertia,
            Err(status) => return status,
        };
        let initial = match check_attitude_state(WHO, "initial", initial) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let seed = match check_attitude_state(WHO, "seed", seed) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let Some(closed) = AttitudeLoop::new(law, latency, step) else {
            return fail(
                SL_ERR_DOMAIN,
                format!(
                    "{WHO}: a step of {step} s must be positive and the latency a whole number of \
                     steps of it"
                ),
            );
        };
        if closed.lag() >= CAPI_LOOP_HISTORY {
            return fail(
                SL_ERR_DOMAIN,
                format!(
                    "{WHO}: a lag of {} steps does not fit this ABI's history of \
                     {CAPI_LOOP_HISTORY}",
                    closed.lag()
                ),
            );
        }

        let mut samples = Vec::with_capacity(steps);
        let final_state = closed.run::<CAPI_LOOP_HISTORY>(
            initial,
            seed,
            &to_rows::<3, 3>(&reference),
            &inertia,
            steps,
            |sample| {
                samples.push(SlLoopSample {
                    time_s: sample.time,
                    error_angle_rad: sample.error.angle(),
                    torque: from_three(sample.torque),
                });
            },
        );
        let Some(final_state) = final_state else {
            return fail(SL_ERR_COMPUTE, format!("{WHO}: the history was outrun"));
        };
        if !out_final.is_null() {
            // SAFETY: checked non-null.
            unsafe {
                *out_final = SlAttitudeState {
                    rotation: from_rows(&final_state.rotation),
                    rate: from_three(final_state.rate),
                };
            }
        }
        // SAFETY: the caller states `out` is null or writable for `cap`, and `out_len` writable.
        unsafe { deliver(&samples, out, cap, out_len) }
    })
}

/// The state an `sl_attitude_state` describes, with its rotation checked finite.
fn check_attitude_state(
    who: &str,
    name: &str,
    state: SlAttitudeState,
) -> Result<AttitudeState, i32> {
    check_all_finite(who, name, &state.rotation)?;
    Ok(AttitudeState::new(
        to_rows::<3, 3>(&state.rotation),
        to_three(state.rate),
    ))
}

/// The inverse of `to_rows::<3, 3>`.
fn from_rows(m: &[[f64; 3]; 3]) -> [f64; 9] {
    core::array::from_fn(|i| m[i / 3][i % 3])
}

/// The delay margin of one linearised axis. See the header.
///
/// # Safety
/// `out` is writable for one `sl_delay_margin`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_delay_margin_of_gains(
    moment: f64,
    angle_gain: f64,
    rate_gain: f64,
    out: *mut SlDelayMargin,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_delay_margin_of_gains";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let Some(axis) = SecondOrderLoop::new(moment, angle_gain, rate_gain) else {
            return fail(
                SL_ERR_DOMAIN,
                format!(
                    "{WHO}: the moment {moment} and the angle gain {angle_gain} must be positive \
                     and the rate gain {rate_gain} non-negative"
                ),
            );
        };
        // SAFETY: checked non-null above.
        unsafe { *out = to_delay_margin(&axis) };
        SL_OK
    })
}

/// The delay margin of a named law. See the header.
///
/// # Safety
/// `out` is writable for one `sl_delay_margin`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_delay_margin_of_law(
    law: i32,
    proportional: f64,
    derivative: f64,
    moment: f64,
    out: *mut SlDelayMargin,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_delay_margin_of_law";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let law = match check_law(WHO, law, proportional, derivative) {
            Ok(law) => law,
            Err(status) => return status,
        };
        let Some(axis) = SecondOrderLoop::of(&law, moment) else {
            return fail(
                SL_ERR_DOMAIN,
                format!("{WHO}: the moment {moment} must be finite and positive"),
            );
        };
        // SAFETY: checked non-null above.
        unsafe { *out = to_delay_margin(&axis) };
        SL_OK
    })
}

/// The binding axis of a three-axis body. See the header.
///
/// # Safety
/// `moments` points at one initialised `sl_vec3`; both out-pointers are writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_worst_principal_axis(
    law: i32,
    proportional: f64,
    derivative: f64,
    moments: *const SlVec3,
    out_axis: *mut usize,
    out: *mut SlDelayMargin,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_worst_principal_axis";
        if moments.is_null() || out_axis.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let law = match check_law(WHO, law, proportional, derivative) {
            Ok(law) => law,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states it is initialised.
        let moments = to_three(unsafe { *moments });
        let Some((axis, margin)) = capi_worst_axis(&law, moments) else {
            return fail(
                SL_ERR_DOMAIN,
                format!("{WHO}: every principal moment must be finite and positive"),
            );
        };
        // The margin is recomputed through the loop so that the four other fields come from the
        // same axis, rather than being reconstructed from a `DelayMargin` that carries only two.
        let Some(worst) = SecondOrderLoop::of(&law, moments[axis]) else {
            return fail(SL_ERR_COMPUTE, format!("{WHO}: the binding axis vanished"));
        };
        let published = to_delay_margin(&worst);
        if published.seconds != margin.seconds() {
            return fail(
                SL_ERR_COMPUTE,
                format!("{WHO}: the binding axis disagrees with its own margin"),
            );
        }
        // SAFETY: both checked non-null above.
        unsafe {
            *out_axis = axis;
            *out = published;
        }
        SL_OK
    })
}

/// The Clohessy-Wiltshire relative plant. See the header.
///
/// # Safety
/// `out_a` is writable for 36 `double` and `out_b` for 18.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_clohessy_wiltshire_plant(
    mean_motion: f64,
    out_a: *mut f64,
    out_b: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_clohessy_wiltshire_plant";
        if out_a.is_null() || out_b.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let Some((a, b)) = capi_cw_plant(mean_motion) else {
            return fail(
                SL_ERR_DOMAIN,
                format!("{WHO}: the mean motion {mean_motion} must be finite and non-zero"),
            );
        };
        // SAFETY: both checked non-null; the caller states the sizes.
        unsafe {
            write_flat(&a, out_a);
            write_flat(&b, out_b);
        }
        SL_OK
    })
}

/// Copy a row-major matrix into a caller's buffer.
///
/// # Safety
/// `out` is writable for `ROWS · COLS` `double`.
unsafe fn write_flat<const COLS: usize, const ROWS: usize>(m: &[[f64; COLS]; ROWS], out: *mut f64) {
    for (i, row) in m.iter().enumerate() {
        // SAFETY: the caller states `out` is writable for `ROWS · COLS`.
        unsafe { core::ptr::copy_nonoverlapping(row.as_ptr(), out.add(i * COLS), COLS) };
    }
}

/// The infinite-horizon LQR gain. See the header.
///
/// # Safety
/// `a` and `q` are readable for `n·n`, `r` for `m·m`, `b` for `n·m`; `out_gain` is null or
/// writable for `m·n`, `out_riccati` null or writable for `n·n`, `out_residual` null or writable
/// for one `double`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn sl_lqr(
    a: *const f64,
    b: *const f64,
    q: *const f64,
    r: *const f64,
    n: usize,
    m: usize,
    out_gain: *mut f64,
    out_riccati: *mut f64,
    out_residual: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_lqr";
        if a.is_null() || b.is_null() || q.is_null() || r.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_lqr_shape(WHO, n, m) {
            return status;
        }
        // SAFETY: checked non-null; the caller states the four sizes.
        let (a, b, q, r) = unsafe {
            (
                core::slice::from_raw_parts(a, n * n),
                core::slice::from_raw_parts(b, n * m),
                core::slice::from_raw_parts(q, n * n),
                core::slice::from_raw_parts(r, m * m),
            )
        };
        let solved = match (n, m) {
            (2, 1) => capi_solve_lqr::<2, 1, 4>(a, b, q, r),
            (3, 1) => capi_solve_lqr::<3, 1, 6>(a, b, q, r),
            (4, 2) => capi_solve_lqr::<4, 2, 8>(a, b, q, r),
            (6, 3) => capi_solve_lqr::<6, 3, 12>(a, b, q, r),
            _ => unreachable!("check_lqr_shape admitted only the four instantiated shapes"),
        };
        let (gain, riccati, residual) = match solved {
            Ok(triple) => triple,
            Err(e) => return fail(SL_ERR_DOMAIN, format!("{WHO}: no optimal gain: {e:?}")),
        };
        if !out_gain.is_null() {
            // SAFETY: checked non-null; the caller states it is writable for `m·n`.
            unsafe { core::ptr::copy_nonoverlapping(gain.as_ptr(), out_gain, gain.len()) };
        }
        if !out_riccati.is_null() {
            // SAFETY: checked non-null; the caller states it is writable for `n·n`.
            unsafe { core::ptr::copy_nonoverlapping(riccati.as_ptr(), out_riccati, riccati.len()) };
        }
        if !out_residual.is_null() {
            // SAFETY: checked non-null.
            unsafe { *out_residual = residual };
        }
        SL_OK
    })
}

/// One of the four `(n, m)` shapes this ABI instantiates the const-generic solver for.
fn check_lqr_shape(who: &str, n: usize, m: usize) -> Result<(), i32> {
    if matches!((n, m), (2, 1) | (3, 1) | (4, 2) | (6, 3)) {
        return Ok(());
    }
    Err(fail(
        SL_ERR_INVALID_ARGUMENT,
        format!("{who}: ({n}, {m}) is not one of the instantiated (2,1), (3,1), (4,2), (6,3)"),
    ))
}

/// The gain, the Riccati solution and the residual, flattened row-major.
fn capi_solve_lqr<const N: usize, const M: usize, const N2: usize>(
    a: &[f64],
    b: &[f64],
    q: &[f64],
    r: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, f64), LqrError> {
    let solved = capi_lqr::<N, M, N2>(
        &to_rows::<N, N>(a),
        &to_rows::<M, N>(b),
        &to_rows::<N, N>(q),
        &to_rows::<M, M>(r),
    )?;
    let flatten = |m: &[&[f64]]| -> Vec<f64> { m.concat() };
    let gain: Vec<&[f64]> = solved.gain().iter().map(|row| &row[..]).collect();
    let riccati: Vec<&[f64]> = solved.riccati().iter().map(|row| &row[..]).collect();
    Ok((flatten(&gain), flatten(&riccati), solved.residual()))
}

/// The station-keeping Lyapunov function. See the header.
///
/// # Safety
/// `law` points at one initialised `sl_element_feedback`; `error` is readable for six `double`;
/// `out` is writable for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_element_feedback_lyapunov(
    law: *const SlElementFeedback,
    error: *const f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_element_feedback_lyapunov";
        if law.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: checked non-null; the caller states it is initialised.
        let law = match check_element_feedback(WHO, unsafe { *law }) {
            Ok(law) => law,
            Err(status) => return status,
        };
        // SAFETY: checked inside; the caller states it is readable for six.
        let error = match unsafe { read_six(WHO, error, "error") } {
            Ok(error) => error,
            Err(status) => return status,
        };
        // SAFETY: checked non-null above.
        unsafe { *out = law.lyapunov(&error) };
        SL_OK
    })
}

/// The station-keeping acceleration. See the header.
///
/// # Safety
/// `law` points at one initialised struct; `elements` and `error` are readable for six `double`;
/// `out` is writable for one `sl_vec3`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_element_feedback_acceleration(
    law: *const SlElementFeedback,
    elements: *const f64,
    mu: f64,
    true_longitude_rad: f64,
    error: *const f64,
    out: *mut SlVec3,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_element_feedback_acceleration";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: the caller states the three pointers are readable at the documented sizes.
        let prepared = unsafe { read_element_feedback(WHO, law, elements, mu, error) };
        let (law, aux, error) = match prepared {
            Ok(triple) => triple,
            Err(status) => return status,
        };
        let command = law.acceleration(&aux, true_longitude_rad, &error);
        // SAFETY: checked non-null above.
        unsafe { *out = from_three(command) };
        SL_OK
    })
}

/// `V̇` along the closed loop. See the header.
///
/// # Safety
/// As `sl_element_feedback_acceleration`, with `out` writable for one `double`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_element_feedback_decrease_rate(
    law: *const SlElementFeedback,
    elements: *const f64,
    mu: f64,
    true_longitude_rad: f64,
    error: *const f64,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_element_feedback_decrease_rate";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: the caller states the three pointers are readable at the documented sizes.
        let prepared = unsafe { read_element_feedback(WHO, law, elements, mu, error) };
        let (law, aux, error) = match prepared {
            Ok(triple) => triple,
            Err(status) => return status,
        };
        // SAFETY: checked non-null above.
        unsafe { *out = law.decrease_rate(&aux, true_longitude_rad, &error) };
        SL_OK
    })
}

/// The three values both station-keeping entry points start from.
///
/// # Safety
/// `law` is null or points at one initialised struct; `elements` and `error` are null or readable
/// for six `double`.
unsafe fn read_element_feedback(
    who: &str,
    law: *const SlElementFeedback,
    elements: *const f64,
    mu: f64,
    error: *const f64,
) -> Result<(ElementFeedback, DsstAuxiliary, [f64; 6]), i32> {
    if law.is_null() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: null pointer"),
        ));
    }
    // SAFETY: checked non-null; the caller states it is initialised.
    let law = check_element_feedback(who, unsafe { *law })?;
    check_positive(who, "mu", mu, SL_ERR_INVALID_ARGUMENT)?;
    // SAFETY: checked inside; the caller states both are readable for six.
    let (elements, error) = unsafe {
        (
            read_six(who, elements, "elements")?,
            read_six(who, error, "error")?,
        )
    };
    check_all_finite(who, "elements", &elements)?;
    Ok((
        law,
        DsstAuxiliary::new(
            elements[0],
            elements[1],
            elements[2],
            elements[3],
            elements[4],
            elements[5],
            mu,
        ),
        error,
    ))
}

/// The Laplacian spectrum of an interaction graph. See the header.
///
/// # Safety
/// `adjacency` is readable for `agents · agents` `double`; `out` is writable for `agents`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_laplacian_spectrum(
    adjacency: *const f64,
    agents: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_laplacian_spectrum";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: the caller states `adjacency` is readable for `agents · agents`.
        let a = match unsafe { read_adjacency(WHO, adjacency, agents) } {
            Ok(a) => a,
            Err(status) => return status,
        };
        fn spectrum<const N: usize>(who: &str, a: &[f64]) -> Result<Vec<f64>, i32> {
            Ok(to_graph::<N>(who, a)?.laplacian_spectrum().to_vec())
        }
        let values = match agents {
            2 => spectrum::<2>(WHO, &a),
            3 => spectrum::<3>(WHO, &a),
            4 => spectrum::<4>(WHO, &a),
            5 => spectrum::<5>(WHO, &a),
            6 => spectrum::<6>(WHO, &a),
            7 => spectrum::<7>(WHO, &a),
            8 => spectrum::<8>(WHO, &a),
            9 => spectrum::<9>(WHO, &a),
            10 => spectrum::<10>(WHO, &a),
            11 => spectrum::<11>(WHO, &a),
            _ => spectrum::<12>(WHO, &a),
        };
        let values = match values {
            Ok(values) => values,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states it is writable for `agents`.
        unsafe { core::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len()) };
        SL_OK
    })
}

/// The adjacency matrix a caller supplied, checked square and finite.
///
/// # Safety
/// `adjacency` is null or readable for `agents · agents` `double`.
unsafe fn read_adjacency(who: &str, adjacency: *const f64, agents: usize) -> Result<Vec<f64>, i32> {
    check_agents(who, agents)?;
    if adjacency.is_null() {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: null adjacency pointer"),
        ));
    }
    // SAFETY: checked non-null; the caller states it is readable for `agents · agents`.
    let a = unsafe { core::slice::from_raw_parts(adjacency, agents * agents) };
    check_all_finite(who, "adjacency", a)?;
    Ok(a.to_vec())
}

/// The algebraic connectivity of an interaction graph. See the header.
///
/// # Safety
/// `adjacency` is readable for `agents · agents` `double`; `out` is writable for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_algebraic_connectivity(
    adjacency: *const f64,
    agents: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_algebraic_connectivity";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: the caller states `adjacency` is readable for `agents · agents`.
        let a = match unsafe { read_adjacency(WHO, adjacency, agents) } {
            Ok(a) => a,
            Err(status) => return status,
        };
        fn connectivity<const N: usize>(who: &str, a: &[f64]) -> Result<f64, i32> {
            Ok(to_graph::<N>(who, a)?.algebraic_connectivity())
        }
        let value = match agents {
            2 => connectivity::<2>(WHO, &a),
            3 => connectivity::<3>(WHO, &a),
            4 => connectivity::<4>(WHO, &a),
            5 => connectivity::<5>(WHO, &a),
            6 => connectivity::<6>(WHO, &a),
            7 => connectivity::<7>(WHO, &a),
            8 => connectivity::<8>(WHO, &a),
            9 => connectivity::<9>(WHO, &a),
            10 => connectivity::<10>(WHO, &a),
            11 => connectivity::<11>(WHO, &a),
            _ => connectivity::<12>(WHO, &a),
        };
        let value = match value {
            Ok(value) => value,
            Err(status) => return status,
        };
        // SAFETY: checked non-null above.
        unsafe { *out = value };
        SL_OK
    })
}

/// The delay bound of a consensus protocol on a graph. See the header.
///
/// # Safety
/// `adjacency` is readable for `agents · agents` `double`; `out` is writable for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_consensus_delay_bound(
    adjacency: *const f64,
    agents: usize,
    protocol: i32,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_consensus_delay_bound";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let protocol = match check_protocol(WHO, protocol) {
            Ok(protocol) => protocol,
            Err(status) => return status,
        };
        // SAFETY: the caller states `adjacency` is readable for `agents · agents`.
        let a = match unsafe { read_adjacency(WHO, adjacency, agents) } {
            Ok(a) => a,
            Err(status) => return status,
        };
        fn bound<const N: usize>(who: &str, a: &[f64], protocol: Protocol) -> Result<f64, i32> {
            let graph = to_graph::<N>(who, a)?;
            // The bound does not read the delay, so any admissible one produces the same answer.
            let net = ConsensusNetwork::new(graph, protocol, 1.0).ok_or_else(|| {
                fail(
                    SL_ERR_COMPUTE,
                    format!("{who}: 1 s was refused as a link delay"),
                )
            })?;
            Ok(net.delay_bound().seconds())
        }
        let value = match agents {
            2 => bound::<2>(WHO, &a, protocol),
            3 => bound::<3>(WHO, &a, protocol),
            4 => bound::<4>(WHO, &a, protocol),
            5 => bound::<5>(WHO, &a, protocol),
            6 => bound::<6>(WHO, &a, protocol),
            7 => bound::<7>(WHO, &a, protocol),
            8 => bound::<8>(WHO, &a, protocol),
            9 => bound::<9>(WHO, &a, protocol),
            10 => bound::<10>(WHO, &a, protocol),
            11 => bound::<11>(WHO, &a, protocol),
            _ => bound::<12>(WHO, &a, protocol),
        };
        let value = match value {
            Ok(value) => value,
            Err(status) => return status,
        };
        // SAFETY: checked non-null above.
        unsafe { *out = value };
        SL_OK
    })
}

/// The disagreement of a state. See the header.
///
/// # Safety
/// `state` is readable for `agents` `double`; `out` is writable for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_consensus_disagreement(
    state: *const f64,
    agents: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_consensus_disagreement";
        if state.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_agents(WHO, agents) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is readable for `agents`.
        let x = unsafe { core::slice::from_raw_parts(state, agents) };
        fn spread<const N: usize>(x: &[f64]) -> Result<f64, i32> {
            Ok(ConsensusNetwork::<N>::disagreement(&to_rows::<N, 1>(x)[0]))
        }
        let value = match agents {
            2 => spread::<2>(x),
            3 => spread::<3>(x),
            4 => spread::<4>(x),
            5 => spread::<5>(x),
            6 => spread::<6>(x),
            7 => spread::<7>(x),
            8 => spread::<8>(x),
            9 => spread::<9>(x),
            10 => spread::<10>(x),
            11 => spread::<11>(x),
            _ => spread::<12>(x),
        };
        let value = match value {
            Ok(value) => value,
            Err(status) => return status,
        };
        // SAFETY: checked non-null above.
        unsafe { *out = value };
        SL_OK
    })
}

/// What one delayed consensus run reports back.
struct ConsensusOutcome {
    final_state: Vec<f64>,
    disagreement: f64,
    accepted: u32,
    rejected: u32,
}

/// Integrate delayed consensus. See the header.
///
/// # Safety
/// `adjacency` is readable for `agents · agents` and `initial` for `agents` `double`; `out_final`
/// is null or writable for `agents`; the three remaining out-pointers are null or writable for
/// one of their type.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn sl_consensus_run(
    adjacency: *const f64,
    agents: usize,
    protocol: i32,
    delay: f64,
    initial: *const f64,
    t1: f64,
    tolerance: f64,
    out_final: *mut f64,
    out_disagreement: *mut f64,
    out_accepted: *mut u32,
    out_rejected: *mut u32,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_consensus_run";
        if initial.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        let protocol = match check_protocol(WHO, protocol) {
            Ok(protocol) => protocol,
            Err(status) => return status,
        };
        if let Err(status) = check_positive(WHO, "delay", delay, SL_ERR_INVALID_ARGUMENT) {
            return status;
        }
        if let Err(status) = check_positive(WHO, "tolerance", tolerance, SL_ERR_INVALID_ARGUMENT) {
            return status;
        }
        if let Err(status) = check_after(WHO, t1, delay) {
            return status;
        }
        // SAFETY: the caller states `adjacency` is readable for `agents · agents`.
        let a = match unsafe { read_adjacency(WHO, adjacency, agents) } {
            Ok(a) => a,
            Err(status) => return status,
        };
        // SAFETY: checked non-null; the caller states it is readable for `agents`.
        let y0 = unsafe { core::slice::from_raw_parts(initial, agents) }.to_vec();
        if let Err(status) = check_all_finite(WHO, "initial", &y0) {
            return status;
        }
        let outcome = match agents {
            2 => run_consensus::<2>(WHO, &a, protocol, delay, &y0, t1, tolerance),
            3 => run_consensus::<3>(WHO, &a, protocol, delay, &y0, t1, tolerance),
            4 => run_consensus::<4>(WHO, &a, protocol, delay, &y0, t1, tolerance),
            5 => run_consensus::<5>(WHO, &a, protocol, delay, &y0, t1, tolerance),
            6 => run_consensus::<6>(WHO, &a, protocol, delay, &y0, t1, tolerance),
            7 => run_consensus::<7>(WHO, &a, protocol, delay, &y0, t1, tolerance),
            8 => run_consensus::<8>(WHO, &a, protocol, delay, &y0, t1, tolerance),
            9 => run_consensus::<9>(WHO, &a, protocol, delay, &y0, t1, tolerance),
            10 => run_consensus::<10>(WHO, &a, protocol, delay, &y0, t1, tolerance),
            11 => run_consensus::<11>(WHO, &a, protocol, delay, &y0, t1, tolerance),
            _ => run_consensus::<12>(WHO, &a, protocol, delay, &y0, t1, tolerance),
        };
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(status) => return status,
        };
        // SAFETY: each checked non-null before it is written; the caller states the sizes.
        unsafe {
            if !out_final.is_null() {
                core::ptr::copy_nonoverlapping(
                    outcome.final_state.as_ptr(),
                    out_final,
                    outcome.final_state.len(),
                );
            }
            if !out_disagreement.is_null() {
                *out_disagreement = outcome.disagreement;
            }
            if !out_accepted.is_null() {
                *out_accepted = outcome.accepted;
            }
            if !out_rejected.is_null() {
                *out_rejected = outcome.rejected;
            }
        }
        SL_OK
    })
}

/// The run must end after it starts, and by at least one delay, or the method of steps has no
/// first arc to take.
fn check_after(who: &str, t1: f64, delay: f64) -> Result<(), i32> {
    if !(t1.is_finite() && t1 > 0.0) {
        return Err(fail(
            SL_ERR_INVALID_ARGUMENT,
            format!("{who}: t1 is {t1}, and the run starts at zero"),
        ));
    }
    if t1 <= delay {
        return Err(fail(
            SL_ERR_DOMAIN,
            format!("{who}: a run to {t1} s is shorter than its own {delay} s delay"),
        ));
    }
    Ok(())
}

/// One delayed consensus run at a fixed agent count.
fn run_consensus<const N: usize>(
    who: &str,
    adjacency: &[f64],
    protocol: Protocol,
    delay: f64,
    initial: &[f64],
    t1: f64,
    tolerance: f64,
) -> Result<ConsensusOutcome, i32> {
    let graph = to_graph::<N>(who, adjacency)?;
    let net = ConsensusNetwork::new(graph, protocol, delay).ok_or_else(|| {
        fail(
            SL_ERR_DOMAIN,
            format!("{who}: the link delay {delay} must be finite and strictly positive"),
        )
    })?;
    let y0 = to_rows::<N, 1>(initial)[0];
    let solution = integrate_delayed(
        &Dop853,
        &net,
        InitialValue::at_rest(0.0, y0),
        t1,
        0.05 * delay,
        Controls::new(Tolerance::new(tolerance, tolerance), 1e-10),
    )
    .map_err(|e| {
        fail(
            SL_ERR_COMPUTE,
            format!("{who}: the delayed run failed: {e:?}"),
        )
    })?;
    let final_state = solution.final_state();
    let (accepted, rejected) = solution.step_counts();
    Ok(ConsensusOutcome {
        final_state: final_state.to_vec(),
        disagreement: ConsensusNetwork::<N>::disagreement(&final_state),
        accepted,
        rejected,
    })
}

/// The propagated discontinuities of a delayed system. See the header.
///
/// # Safety
/// `delays` is readable for `n` `double`; `out` is null or writable for `cap`; `out_len` is
/// writable for one `size_t`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn sl_delay_breakpoints(
    delays: *const f64,
    n: usize,
    t0: f64,
    t1: f64,
    levels: usize,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_delay_breakpoints";
        if delays.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_nonzero(WHO, "n", n) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is readable for `n`.
        let delays = unsafe { core::slice::from_raw_parts(delays, n) };
        let Some(values) = breakpoints(delays, t0, t1, levels) else {
            return fail(
                SL_ERR_DOMAIN,
                format!(
                    "{WHO}: every delay must be finite and positive, t1 ({t1}) must follow t0 \
                     ({t0}), and the breakpoint count must stay inside its cap"
                ),
            );
        };
        // SAFETY: the caller states `out` is null or writable for `cap`, and `out_len` writable.
        unsafe { deliver(&values, out, cap, out_len) }
    })
}

/// The error quaternion the same two frames describe. See the header.
///
/// # Safety
/// As `sl_attitude_error_mrp`, plus `out` writable for four `double`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_attitude_error_quaternion(
    rotation: *const f64,
    reference: *const f64,
    rate: *const SlVec3,
    rate_reference: *const SlVec3,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_attitude_error_quaternion";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: the caller states the four inputs are readable at the documented sizes.
        let frames = unsafe { read_attitude_pair(WHO, rotation, reference, rate, rate_reference) };
        let error = match frames {
            Ok(error) => error,
            Err(status) => return status,
        };
        let q = error.quaternion();
        // SAFETY: checked non-null; the caller states it is writable for four.
        unsafe { core::ptr::copy_nonoverlapping(q.as_ptr(), out, 4) };
        SL_OK
    })
}

/// The Laplacian eigenvalues and their eigenvectors. See the header.
///
/// # Safety
/// `adjacency` is readable for `agents · agents`; `out_values` is null or writable for `agents`,
/// and `out_modes` null or writable for `agents · agents`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_laplacian_modes(
    adjacency: *const f64,
    agents: usize,
    out_values: *mut f64,
    out_modes: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_laplacian_modes";
        // SAFETY: the caller states `adjacency` is readable for `agents · agents`.
        let a = match unsafe { read_adjacency(WHO, adjacency, agents) } {
            Ok(a) => a,
            Err(status) => return status,
        };
        fn eigen<const N: usize>(who: &str, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32> {
            let (values, modes) = to_graph::<N>(who, a)?.laplacian_eigen();
            Ok((values.to_vec(), modes.concat()))
        }
        let pair = match agents {
            2 => eigen::<2>(WHO, &a),
            3 => eigen::<3>(WHO, &a),
            4 => eigen::<4>(WHO, &a),
            5 => eigen::<5>(WHO, &a),
            6 => eigen::<6>(WHO, &a),
            7 => eigen::<7>(WHO, &a),
            8 => eigen::<8>(WHO, &a),
            9 => eigen::<9>(WHO, &a),
            10 => eigen::<10>(WHO, &a),
            11 => eigen::<11>(WHO, &a),
            _ => eigen::<12>(WHO, &a),
        };
        let (values, modes) = match pair {
            Ok(pair) => pair,
            Err(status) => return status,
        };
        // SAFETY: each checked non-null before it is written; the caller states the sizes.
        unsafe {
            if !out_values.is_null() {
                core::ptr::copy_nonoverlapping(values.as_ptr(), out_values, values.len());
            }
            if !out_modes.is_null() {
                core::ptr::copy_nonoverlapping(modes.as_ptr(), out_modes, modes.len());
            }
        }
        SL_OK
    })
}

/// Whether the interaction graph is connected. See the header.
///
/// # Safety
/// `adjacency` is readable for `agents · agents`; `out` is writable for one `bool`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_graph_is_connected(
    adjacency: *const f64,
    agents: usize,
    out: *mut bool,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_graph_is_connected";
        if out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        // SAFETY: the caller states `adjacency` is readable for `agents · agents`.
        let a = match unsafe { read_adjacency(WHO, adjacency, agents) } {
            Ok(a) => a,
            Err(status) => return status,
        };
        // `Graph::new` refuses a disconnected graph, and `is_connected` is only reachable
        // through it — so the answer this entry point exists to give arrives as that refusal, and
        // reading it is the whole of the function. Every other `GraphError` stays a refusal: a
        // matrix that is not symmetric is not a disconnected graph, it is not a graph.
        fn connected<const N: usize>(who: &str, a: &[f64]) -> Result<bool, i32> {
            match Graph::<N>::new(to_rows::<N, N>(a)) {
                Ok(_) => Ok(true),
                Err(GraphError::Disconnected) => Ok(false),
                Err(e) => Err(fail(
                    SL_ERR_DOMAIN,
                    format!("{who}: not a usable interaction graph: {e:?}"),
                )),
            }
        }
        let verdict = match agents {
            2 => connected::<2>(WHO, &a),
            3 => connected::<3>(WHO, &a),
            4 => connected::<4>(WHO, &a),
            5 => connected::<5>(WHO, &a),
            6 => connected::<6>(WHO, &a),
            7 => connected::<7>(WHO, &a),
            8 => connected::<8>(WHO, &a),
            9 => connected::<9>(WHO, &a),
            10 => connected::<10>(WHO, &a),
            11 => connected::<11>(WHO, &a),
            _ => connected::<12>(WHO, &a),
        };
        let verdict = match verdict {
            Ok(verdict) => verdict,
            Err(status) => return status,
        };
        // SAFETY: checked non-null above.
        unsafe { *out = verdict };
        SL_OK
    })
}

/// The monic characteristic polynomial of a square matrix. See the header.
///
/// # Safety
/// `m` is readable for `n · n`; `out` is writable for `n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sl_characteristic_polynomial(
    m: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    guard(|| {
        const WHO: &str = "sl_characteristic_polynomial";
        if m.is_null() || out.is_null() {
            return fail(SL_ERR_INVALID_ARGUMENT, format!("{WHO}: null pointer"));
        }
        if let Err(status) = check_matrix_order(WHO, n) {
            return status;
        }
        // SAFETY: checked non-null; the caller states it is readable for `n · n`.
        let entries = unsafe { core::slice::from_raw_parts(m, n * n) };
        if let Err(status) = check_all_finite(WHO, "m", entries) {
            return status;
        }
        fn coefficients<const N: usize>(m: &[f64]) -> Vec<f64> {
            control_characteristic_polynomial(&to_rows::<N, N>(m)).to_vec()
        }
        let c = match n {
            2 => coefficients::<2>(entries),
            3 => coefficients::<3>(entries),
            4 => coefficients::<4>(entries),
            5 => coefficients::<5>(entries),
            6 => coefficients::<6>(entries),
            7 => coefficients::<7>(entries),
            8 => coefficients::<8>(entries),
            9 => coefficients::<9>(entries),
            10 => coefficients::<10>(entries),
            11 => coefficients::<11>(entries),
            _ => coefficients::<12>(entries),
        };
        // SAFETY: checked non-null; the caller states it is writable for `n`.
        unsafe { core::ptr::copy_nonoverlapping(c.as_ptr(), out, c.len()) };
        SL_OK
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use spacelab_estimation::SOLAR_RADIUS;

    /// ISS, epoch 2008-264 — the element set the `spacelab-sgp4` suite is written against, reused
    /// here so a disagreement is about the binding rather than about which TLE was meant.
    const LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    const LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    /// The header, read at compile time so the constant test cannot be fooled by a stale copy.
    const HEADER: &str = include_str!("../include/spacelab.h");

    /// Earth's second zonal harmonic, the value `SL_J2_EARTH` must equal.
    const J2_EARTH: f64 = 1.082_626_68e-3;

    /// Earth's third zonal harmonic, the value `SL_J3_EARTH` must equal.
    const J3_EARTH: f64 = -2.532_653_65e-6;

    /// The literal a `#define NAME <literal>` gives, or a failure naming the macro that is missing.
    fn define(name: &str) -> &'static str {
        HEADER
            .lines()
            .find_map(|line| line.strip_prefix(&format!("#define {name} ")))
            .unwrap_or_else(|| panic!("{name} is not defined in spacelab.h"))
            .trim()
    }

    fn define_f64(name: &str) -> f64 {
        // A negative replacement list is parenthesised in the header, so that it cannot bind
        // wrongly at a use site; the parentheses are punctuation, not part of the value.
        define(name)
            .trim_start_matches('(')
            .trim_end_matches(')')
            .parse()
            .unwrap_or_else(|e| panic!("{name} is not a floating-point literal: {e}"))
    }

    fn define_i32(name: &str) -> i32 {
        define(name)
            .trim_end_matches(['u', 'U'])
            .parse()
            .unwrap_or_else(|e| panic!("{name} is not an integer literal: {e}"))
    }

    fn state(r: [f64; 3], v: [f64; 3]) -> SlState {
        SlState {
            position: SlVec3 {
                x: r[0],
                y: r[1],
                z: r[2],
            },
            velocity: SlVec3 {
                x: v[0],
                y: v[1],
                z: v[2],
            },
        }
    }

    /// A low circular orbit at the ISS inclination, used wherever a test needs a state that is not
    /// degenerate in any of the terms a frame transform or a topocentric rotation multiplies.
    fn leo() -> SlState {
        let a = 6_778_137.0;
        let speed = (MU_EARTH / a).sqrt();
        let inc: f64 = 51.6_f64.to_radians();
        state([a, 0.0, 0.0], [0.0, speed * inc.cos(), speed * inc.sin()])
    }

    fn last_error() -> String {
        // SAFETY: a buffer this function owns, sized from the library's own answer.
        let needed = unsafe { sl_last_error_message(core::ptr::null_mut(), 0) };
        let mut buf = vec![0_u8; needed];
        // SAFETY: `buf` is writable for exactly `needed` bytes.
        unsafe { sl_last_error_message(buf.as_mut_ptr().cast::<c_char>(), needed) };
        buf.pop();
        String::from_utf8(buf).expect("the library writes UTF-8")
    }

    /// Every `#define` in the header is the value the library computes with. A header that drifts
    /// from the code is worse than no header: it compiles, it links, and it is wrong by whatever
    /// the drift is — here, a truncated μ would move a Walker constellation without any test in
    /// the workspace noticing, because nothing else reads the header at all.
    #[test]
    fn the_header_constants_are_the_values_the_library_computes_with() {
        assert_eq!(
            define_i32("SL_ABI_VERSION"),
            i32::try_from(ABI_VERSION).expect("fits")
        );

        assert_eq!(
            define_f64("SL_MU_EARTH").to_bits(),
            MU_EARTH.to_bits(),
            "SL_MU_EARTH"
        );
        assert_eq!(
            define_f64("SL_RE_EARTH").to_bits(),
            RE_EARTH.to_bits(),
            "SL_RE_EARTH"
        );
        assert_eq!(
            define_f64("SL_J2_EARTH").to_bits(),
            J2_EARTH.to_bits(),
            "SL_J2_EARTH"
        );
        assert_eq!(
            define_f64("SL_J3_EARTH").to_bits(),
            J3_EARTH.to_bits(),
            "SL_J3_EARTH"
        );

        // The rotation rate has an authoritative home in `spacelab-frames`, so the header is
        // checked against that rather than against a second copy written here. Its Python-side
        // sibling is truncated at `7.292_115_146_7e-5`; this one is not.
        assert_eq!(
            define_f64("SL_EARTH_ROTATION_RATE").to_bits(),
            spacelab_frames::EARTH_ROTATION_RATE.to_bits(),
            "SL_EARTH_ROTATION_RATE",
        );
        assert_eq!(
            define_f64("SL_RE_EARTH").to_bits(),
            spacelab_earth::EQUATORIAL_RADIUS.to_bits(),
            "SL_RE_EARTH against the WGS-84 crate constant",
        );

        for (name, value) in [
            ("SL_OK", SL_OK),
            ("SL_ERR_INVALID_ARGUMENT", SL_ERR_INVALID_ARGUMENT),
            ("SL_ERR_DOMAIN", SL_ERR_DOMAIN),
            ("SL_ERR_PARSE", SL_ERR_PARSE),
            ("SL_ERR_COMPUTE", SL_ERR_COMPUTE),
            ("SL_ERR_BUFFER_TOO_SMALL", SL_ERR_BUFFER_TOO_SMALL),
            ("SL_ERR_PANIC", SL_ERR_PANIC),
            ("SL_ERR_UNSUPPORTED", SL_ERR_UNSUPPORTED),
        ] {
            assert_eq!(define_i32(name), value, "{name}");
        }

        // `sl_tle`'s two fields are sized by this macro, so a header that disagreed with the
        // library would have the caller reading one byte past a line or stopping one short of its
        // terminator -- neither of which the compiler can see across the ABI.
        assert_eq!(
            usize::try_from(define_i32("SL_TLE_LINE_BYTES")).expect("a positive length"),
            TLE_LINE_BYTES,
            "SL_TLE_LINE_BYTES"
        );

        assert_eq!(sl_abi_version(), ABI_VERSION);
    }

    /// The header declares the same enumerators the library matches on, and the same struct field
    /// order. Order is the part no compiler checks across the boundary: swap two `double` fields in
    /// `sl_soc_geometry` and both sides still build, both still link, and every angle is somebody
    /// else's.
    #[test]
    fn the_header_tags_and_layouts_are_the_ones_the_library_uses() {
        for (name, value) in [
            ("SL_PRECISION_DOUBLE", PRECISION_DOUBLE),
            ("SL_PRECISION_DOUBLE_DOUBLE", PRECISION_DOUBLE_DOUBLE),
            ("SL_WALKER_DELTA", WALKER_DELTA),
            ("SL_WALKER_STAR", WALKER_STAR),
            ("SL_BAND_S", BAND_S),
            ("SL_BAND_X", BAND_X),
            ("SL_BAND_KA", BAND_KA),
            ("SL_LIFETIME_REENTRY", LIFETIME_REENTRY),
            ("SL_LIFETIME_SURVIVES", LIFETIME_SURVIVES),
            ("SL_LIFETIME_UNRESOLVED", LIFETIME_UNRESOLVED),
            ("SL_SLIP_ARC_START", SLIP_ARC_START),
            ("SL_SLIP_CONTINUOUS", SLIP_CONTINUOUS),
            ("SL_SLIP_SLIPPED", SLIP_SLIPPED),
        ] {
            let assigned = HEADER
                .split_once(&format!("{name} = "))
                .unwrap_or_else(|| panic!("{name} is not an enumerator in spacelab.h"))
                .1;
            let assigned: i32 = assigned
                .split(|c: char| !c.is_ascii_digit() && c != '-')
                .next()
                .and_then(|t| t.parse().ok())
                .unwrap_or_else(|| panic!("{name} has no integer value in spacelab.h"));
            assert_eq!(assigned, value, "{name}");
        }

        // The enumerators that pin each enum to four bytes. Deleting one compiles, passes every
        // other test, and silently changes the ABI for any consumer built with `-fshort-enums` --
        // which is the default on most ARM bare-metal toolchains. Measured: without them
        // `sizeof(sl_precision)` falls to 1 under that flag.
        for pin in [
            "SL_PRECISION_FORCE_INT32",
            "SL_WALKER_PATTERN_FORCE_INT32",
            "SL_SENSOR_KIND_FORCE_INT32",
            "SL_BAND_FORCE_INT32",
            "SL_LIFETIME_OUTCOME_FORCE_INT32",
            "SL_SLIP_FORCE_INT32",
        ] {
            assert!(
                HEADER.contains(&format!("{pin} = 0x7FFFFFFF")),
                "{pin} is missing from spacelab.h, so the enum width is the compiler's choice"
            );
        }

        // Field order, read out of the header's own struct bodies. `sl_state` is two `sl_vec3`,
        // which is six packed doubles, which is what `to_six` hands the measurement models -- so
        // this also pins the order that function relies on.
        for (name, fields) in [
            ("sl_vec3", &["x", "y", "z"][..]),
            ("sl_state", &["position", "velocity"][..]),
            (
                "sl_geodetic",
                &["latitude_rad", "longitude_rad", "height_m"][..],
            ),
            (
                "sl_soc_geometry",
                &[
                    "planes",
                    "per_plane",
                    "footprint_half_angle_rad",
                    "street_half_width_rad",
                    "raan_step_co_rad",
                    "raan_step_counter_rad",
                    "raan_step_rad",
                    "raan_span_rad",
                    "inter_plane_phase_rad",
                ][..],
            ),
            ("sl_terminal", &["position", "velocity", "epoch"][..]),
            (
                "sl_ramp_segment",
                &["start", "end", "start_frequency", "rate"][..],
            ),
            (
                "sl_doppler_link",
                &[
                    "uplink",
                    "downlink",
                    "reference_frequency",
                    "reference_band",
                    "count_interval",
                    "subtracted",
                ][..],
            ),
            ("sl_corona_term", &["coefficient", "exponent"][..]),
            (
                "sl_decay_params",
                &[
                    "cd",
                    "area_over_mass",
                    "reference_density",
                    "reference_radius",
                    "scale_height",
                    "horizon",
                    "chunk",
                    "tolerance",
                    "density_scale",
                    "mu",
                    "body_radius",
                    "reentry_altitude",
                    "rotation_rate",
                ][..],
            ),
            (
                "sl_lifetime",
                &["outcome", "seconds", "perigee_altitude_m"][..],
            ),
            (
                "sl_lifetime_ensemble",
                &["members", "survivors", "unresolved"][..],
            ),
            (
                "sl_lanes",
                &["lambda1", "lambda2", "wide_lane", "narrow_lane"][..],
            ),
            ("sl_slip_verdict", &["verdict", "statistic"][..]),
            (
                "sl_conjunction",
                &[
                    "relative_position",
                    "relative_velocity",
                    "primary_covariance[9]",
                    "secondary_covariance[9]",
                    "primary_radius",
                    "secondary_radius",
                ][..],
            ),
            (
                "sl_probability_estimates",
                &["foster", "patera", "chan", "maximum"][..],
            ),
            (
                "sl_relative_sample",
                &["position", "velocity", "covariance[9]"][..],
            ),
            (
                "sl_usage_violations",
                &[
                    "any",
                    "non_positive_definite",
                    "offset",
                    "extended",
                    "inaccuracy",
                    "offset_indicator",
                    "extended_indicator",
                    "inaccuracy_indicator",
                    "correction_factor",
                ][..],
            ),
            (
                "sl_close_approach",
                &["tca", "distance", "relative_speed"][..],
            ),
        ] {
            let body = HEADER
                .split_once(&format!("}} {name};"))
                .unwrap_or_else(|| panic!("{name} is not a struct in spacelab.h"))
                .0;
            let body = body
                .rsplit_once("typedef struct {")
                .expect("a struct body")
                .1;
            let declared: Vec<&str> = body
                .lines()
                // A field may carry a trailing `/* unit */`, which documents it rather than
                // declaring it.
                .map(|l| l.split_once("/*").map_or(l, |(code, _)| code).trim())
                .filter_map(|l| l.strip_suffix(';'))
                .filter_map(|l| l.rsplit_once(' '))
                .map(|(_, field)| field)
                .collect();
            assert_eq!(declared, fields, "{name} field order");
        }

        assert_eq!(
            size_of::<SlState>(),
            6 * size_of::<f64>(),
            "sl_state must be six packed f64"
        );
        assert_eq!(align_of::<SlState>(), align_of::<f64>());
    }

    /// The frame entry points are their crate transforms and nothing else. Checked bit-for-bit
    /// against a direct call, because every plausible wiring mistake here -- swapping UT1 for TT,
    /// dropping the length-of-day, passing polar motion in the wrong slot -- produces a number that
    /// is merely a little wrong, and a tolerance would hide all three.
    #[test]
    fn the_frame_bindings_are_their_crate_transforms() {
        let s = leo();
        let (ut1_ns, tt_ns) = (813_456_789_012_345_678_i64, 813_456_789_081_184_678_i64);
        let (xp, yp, dx, dy, lod) = (1.2e-7, -3.4e-7, 5.6e-10, -7.8e-10, 1.5e-3);

        let mut teme = state([0.0; 3], [0.0; 3]);
        // SAFETY: both pointers are to locals of the right type.
        assert_eq!(
            unsafe { sl_teme_to_itrf(&raw const s, ut1_ns, tt_ns, xp, yp, lod, &raw mut teme) },
            SL_OK
        );
        let expect = frames_teme_to_itrf(
            Epoch::<Ut1>::from_ns_since_j2000(i128::from(ut1_ns)),
            Epoch::<Tt>::from_ns_since_j2000(i128::from(tt_ns)),
            Angle::from_radians(xp),
            Angle::from_radians(yp),
            lod,
        )
        .state(StateVector::<Teme>::new(
            to_vec3(s.position),
            to_vec3(s.velocity),
        ));
        assert_eq!(
            teme,
            from_state(expect.position.vector, expect.velocity.vector)
        );

        let mut gcrf = state([0.0; 3], [0.0; 3]);
        // SAFETY: both pointers are to locals of the right type.
        let status = unsafe {
            sl_gcrf_to_itrf(
                &raw const s,
                ut1_ns,
                tt_ns,
                xp,
                yp,
                dx,
                dy,
                lod,
                &raw mut gcrf,
            )
        };
        assert_eq!(status, SL_OK);
        let expect = frames_gcrf_to_itrf(
            Epoch::<Ut1>::from_ns_since_j2000(i128::from(ut1_ns)),
            Epoch::<Tt>::from_ns_since_j2000(i128::from(tt_ns)),
            Angle::from_radians(xp),
            Angle::from_radians(yp),
            Angle::from_radians(dx),
            Angle::from_radians(dy),
            lod,
        )
        .state(StateVector::<Gcrf>::new(
            to_vec3(s.position),
            to_vec3(s.velocity),
        ));
        assert_eq!(
            gcrf,
            from_state(expect.position.vector, expect.velocity.vector)
        );

        // The two chains disagree, and the size of the disagreement is itself a check. Measured at
        // 39 072 m on this epoch, which is 5.7 mrad at this radius, or 0.33 degrees. General
        // precession is 50.3 arcsec a year and this epoch is 25.8 years past J2000, so the
        // accumulated angle is 0.36 degrees: the gap *is* precession, which is exactly what TEME's
        // fixed axis does not model and the CIO chain does. Identical would mean one binding wired
        // to both; an order out would mean an epoch dropped on one side.
        let separation =
            (teme.position.x - gcrf.position.x).hypot(teme.position.y - gcrf.position.y);
        assert!(
            (1.0e4..1.0e5).contains(&separation),
            "TEME and GCRF chains are {separation} m apart, which is not the precession between \
             them at this epoch",
        );
    }

    /// Geodesy round-trips, and each direction is its crate function. The round trip alone would
    /// pass with both directions wrong in the same way -- degrees in, degrees out -- so the
    /// forward leg is also pinned against a hand-checked latitude.
    #[test]
    fn the_geodetic_bindings_invert_each_other_and_their_crate_functions() {
        let g = SlGeodetic {
            latitude_rad: 35.0_f64.to_radians(),
            longitude_rad: 139.0_f64.to_radians(),
            height_m: 42.0,
        };
        let mut r = SlVec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        // SAFETY: both pointers are to locals of the right type.
        assert_eq!(
            unsafe { sl_geodetic_to_itrf(&raw const g, &raw mut r) },
            SL_OK
        );

        let expect = earth_geodetic_to_itrf(Geodetic {
            latitude: Angle::from_radians(g.latitude_rad),
            longitude: Angle::from_radians(g.longitude_rad),
            height: Length::from_metres(g.height_m),
        });
        assert_eq!(r, from_vec3(expect.vector));

        let mut back = SlGeodetic {
            latitude_rad: 0.0,
            longitude_rad: 0.0,
            height_m: 0.0,
        };
        // SAFETY: both pointers are to locals of the right type.
        assert_eq!(
            unsafe { sl_itrf_to_geodetic(&raw const r, &raw mut back) },
            SL_OK
        );
        assert!(
            (back.latitude_rad - g.latitude_rad).abs() < 1e-12,
            "latitude round trip"
        );
        assert!(
            (back.longitude_rad - g.longitude_rad).abs() < 1e-12,
            "longitude round trip"
        );
        assert!(
            (back.height_m - g.height_m).abs() < 1e-6,
            "height round trip"
        );

        // Radians, not degrees. Measured: 35 degrees north on the ellipsoid is z = 3 637 891.0 m.
        // Read the same field as degrees it would be 35 radians, which wraps to 0.61 rad and puts
        // the point at a different latitude entirely -- so the discriminator is not the order of
        // magnitude but the value, and it is checked against the measurement rather than a guess.
        assert!(
            (r.z - 3_637_891.0).abs() < 1.0,
            "z was {} m, measured 3 637 891.0",
            r.z
        );

        // The 43 km geocentric singularity is the one input with no answer, and it must arrive as
        // a named failure rather than as a NaN the caller propagates into an orbit.
        let inside = SlVec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        // SAFETY: both pointers are to locals of the right type.
        assert_eq!(
            unsafe { sl_itrf_to_geodetic(&raw const inside, &raw mut back) },
            SL_ERR_COMPUTE
        );
        assert!(
            last_error().contains("sl_itrf_to_geodetic"),
            "{}",
            last_error()
        );
    }

    /// The SGP4 handle propagates what the satellite does, the batch agrees with the singles, and
    /// the one-shot form agrees with the handle. A malformed element set is a parse failure, not a
    /// silent zero state.
    #[test]
    fn the_sgp4_bindings_propagate_what_the_satellite_does() {
        let mut handle: *mut SlSgp4 = core::ptr::null_mut();
        // SAFETY: NUL-terminated literals and a local out pointer.
        let status = unsafe { sl_sgp4_open(c"...".as_ptr(), c"...".as_ptr(), &raw mut handle) };
        assert_eq!(
            status, SL_ERR_PARSE,
            "a three-dot TLE is not an element set"
        );
        assert!(
            handle.is_null(),
            "a failed open must not hand back a handle"
        );

        let l1 = std::ffi::CString::new(LINE1).expect("no interior NUL");
        let l2 = std::ffi::CString::new(LINE2).expect("no interior NUL");
        // SAFETY: NUL-terminated strings and a local out pointer.
        assert_eq!(
            unsafe { sl_sgp4_open(l1.as_ptr(), l2.as_ptr(), &raw mut handle) },
            SL_OK
        );
        assert!(!handle.is_null());

        let reference = Satellite::from_tle(LINE1, LINE2).expect("the fixture parses");
        let offsets = [0.0, 45.0, 90.0, -30.0];

        let mut singles = Vec::new();
        for &m in &offsets {
            let mut got = state([0.0; 3], [0.0; 3]);
            // SAFETY: a live handle and a local out pointer.
            assert_eq!(
                unsafe { sl_sgp4_propagate_at(handle, m, &raw mut got) },
                SL_OK
            );
            let want = reference
                .propagate_minutes(m)
                .expect("the reference propagates");
            assert_eq!(
                got,
                from_state(want.position.vector, want.velocity.vector),
                "at {m} min"
            );
            singles.push(got);
        }

        let mut batch = vec![state([0.0; 3], [0.0; 3]); offsets.len()];
        // SAFETY: a live handle, `offsets.len()` doubles in, and a distinct buffer of the same
        // length out.
        let status = unsafe {
            sl_sgp4_propagate_batch(handle, offsets.as_ptr(), offsets.len(), batch.as_mut_ptr())
        };
        assert_eq!(status, SL_OK);
        assert_eq!(batch, singles, "the batch must be the singles, parsed once");

        let mut one_shot = state([0.0; 3], [0.0; 3]);
        // SAFETY: NUL-terminated strings and a local out pointer.
        let status =
            unsafe { sl_sgp4_propagate(l1.as_ptr(), l2.as_ptr(), 45.0, &raw mut one_shot) };
        assert_eq!(status, SL_OK);
        assert_eq!(
            one_shot, singles[1],
            "the one-shot form must be open + propagate + close"
        );

        // SAFETY: the handle from `sl_sgp4_open`, closed exactly once.
        unsafe { sl_sgp4_close(handle) };
        // SAFETY: documented as a no-op, and the only way to write cleanup that cannot double-free.
        unsafe { sl_sgp4_close(core::ptr::null_mut()) };
    }

    /// A TLE line with no terminator inside the bound is refused rather than scanned onward. The
    /// buffer here is deliberately *not* NUL-terminated, which is the mistake this bound exists for.
    #[test]
    fn an_unterminated_tle_line_is_refused_rather_than_scanned_past() {
        let unterminated = vec![b'1'; MAX_C_STRING + 1];
        let l2 = std::ffi::CString::new(LINE2).expect("no interior NUL");
        let mut handle: *mut SlSgp4 = core::ptr::null_mut();
        // SAFETY: the first argument is readable for `MAX_C_STRING + 1` bytes, which is one more
        // than the scan will touch, so the over-read the bound prevents cannot happen here either.
        let status = unsafe {
            sl_sgp4_open(
                unterminated.as_ptr().cast::<c_char>(),
                l2.as_ptr(),
                &raw mut handle,
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
        assert!(handle.is_null());
        assert!(last_error().contains("unterminated"), "{}", last_error());
    }

    /// The two-body bindings: the analytic solution and the symplectic one agree over a short arc,
    /// the double-double arm beats the `f64` arm on a time-symmetric round trip, and a precision
    /// tag that is neither is rejected instead of silently taken as `f64`.
    #[test]
    fn the_two_body_bindings_agree_with_kepler_and_with_each_other() {
        let s = leo();
        let dt = 60.0;
        let steps = 90_u64;

        let mut analytic = state([0.0; 3], [0.0; 3]);
        // SAFETY: both pointers are to locals of the right type.
        let status = unsafe {
            sl_kepler_propagate(&raw const s, MU_EARTH, dt * steps as f64, &raw mut analytic)
        };
        assert_eq!(status, SL_OK);
        let (r, v) = astro_kepler(
            to_vec3(s.position),
            to_vec3(s.velocity),
            MU_EARTH,
            dt * steps as f64,
        );
        assert_eq!(
            analytic,
            from_state(r, v),
            "the Kepler binding is its crate function"
        );

        let mut numeric = state([0.0; 3], [0.0; 3]);
        // SAFETY: both pointers are to locals of the right type.
        let status = unsafe {
            sl_propagate_two_body(
                &raw const s,
                MU_EARTH,
                dt,
                steps,
                PRECISION_DOUBLE,
                &raw mut numeric,
            )
        };
        assert_eq!(status, SL_OK);
        // Measured separation over 90 minutes at a 60 s step: 8.0e-5 m. The bound is loose enough
        // to survive a compiler change and tight enough that a dropped step or a doubled one shows.
        let gap = (numeric.position.x - analytic.position.x)
            .hypot(numeric.position.y - analytic.position.y)
            .hypot(numeric.position.z - analytic.position.z);
        assert!(
            gap < 1.0,
            "symplectic and analytic differ by {gap} m over one revolution"
        );

        // Forward then backward with the step negated. Yoshida8 is time-symmetric, so the only
        // thing between the start and the return is round-off -- which is exactly what the two
        // precisions differ in, and the only discriminator that tells them apart at all.
        let mut round_trip = [0.0_f64; 2];
        for (i, precision) in [PRECISION_DOUBLE, PRECISION_DOUBLE_DOUBLE]
            .into_iter()
            .enumerate()
        {
            let mut there = state([0.0; 3], [0.0; 3]);
            let mut back = state([0.0; 3], [0.0; 3]);
            // SAFETY: all pointers are to locals of the right type.
            unsafe {
                assert_eq!(
                    sl_propagate_two_body(
                        &raw const s,
                        MU_EARTH,
                        dt,
                        steps,
                        precision,
                        &raw mut there
                    ),
                    SL_OK
                );
                assert_eq!(
                    sl_propagate_two_body(
                        &raw const there,
                        MU_EARTH,
                        -dt,
                        steps,
                        precision,
                        &raw mut back
                    ),
                    SL_OK
                );
            }
            round_trip[i] = (back.position.x - s.position.x)
                .hypot(back.position.y - s.position.y)
                .hypot(back.position.z - s.position.z);
        }
        // Measured: 2.93e-7 m in f64 against 8.63e-10 m in double-double, a factor of 340.
        //
        // Not the factor of thousands the same experiment shows inside Rust, and the reason is the
        // boundary rather than the arithmetic: this ABI passes `double`, so the state is truncated
        // to f64 between the outbound call and the return one. Double-double buys accuracy *within*
        // a call and cannot carry any across one. A caller who wants the full benefit has to ask
        // for the whole span in a single call -- which is worth knowing, and is the sort of thing
        // an API only tells you if somebody measures it.
        assert!(
            round_trip[1] * 10.0 < round_trip[0],
            "double-double returned {} m from the round trip against f64's {} m, which is not the \
             separation two different precisions produce",
            round_trip[1],
            round_trip[0],
        );

        let mut ignored = state([0.0; 3], [0.0; 3]);
        // SAFETY: both pointers are to locals of the right type.
        let status = unsafe {
            sl_propagate_two_body(&raw const s, MU_EARTH, dt, steps, 7, &raw mut ignored)
        };
        assert_eq!(
            status, SL_ERR_INVALID_ARGUMENT,
            "an unknown precision tag must not default"
        );
        assert!(last_error().contains("precision tag 7"), "{}", last_error());
    }

    /// Lambert returns velocities that actually fly the transfer -- checked by flying it -- and
    /// says so in the domain when no arc exists rather than returning a zero velocity.
    #[test]
    fn the_lambert_binding_returns_velocities_that_fly_the_transfer() {
        let start = leo();
        let tof = 1_800.0;
        let mut arrival = state([0.0; 3], [0.0; 3]);
        // SAFETY: both pointers are to locals of the right type.
        assert_eq!(
            unsafe { sl_kepler_propagate(&raw const start, MU_EARTH, tof, &raw mut arrival) },
            SL_OK
        );

        let (mut v1, mut v2) = (
            SlVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            SlVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        );
        // SAFETY: all four pointers are to locals of the right type.
        let status = unsafe {
            sl_lambert(
                &raw const start.position,
                &raw const arrival.position,
                tof,
                MU_EARTH,
                true,
                &raw mut v1,
                &raw mut v2,
            )
        };
        assert_eq!(status, SL_OK);

        // The transfer between two points of one orbit is that orbit, so the departure velocity
        // Lambert returns must be the one the state already had. This is the self-checking form:
        // the answer is fed back into the question rather than compared with a copied constant.
        for (got, want) in [
            (v1.x, start.velocity.x),
            (v1.y, start.velocity.y),
            (v1.z, start.velocity.z),
        ] {
            assert!(
                (got - want).abs() < 1e-6,
                "departure velocity {got} against {want}"
            );
        }
        for (got, want) in [
            (v2.x, arrival.velocity.x),
            (v2.y, arrival.velocity.y),
            (v2.z, arrival.velocity.z),
        ] {
            assert!(
                (got - want).abs() < 1e-6,
                "arrival velocity {got} against {want}"
            );
        }

        // A transfer that no conic covers: the same two points in no time at all.
        // SAFETY: all four pointers are to locals of the right type.
        let status = unsafe {
            sl_lambert(
                &raw const start.position,
                &raw const arrival.position,
                0.0,
                MU_EARTH,
                true,
                &raw mut v1,
                &raw mut v2,
            )
        };
        assert_eq!(status, SL_ERR_DOMAIN);
        assert!(last_error().contains("no transfer arc"), "{}", last_error());
    }

    /// The four measurement bindings are their crate functions, and a null pointer gives the NaN
    /// the header promises rather than a zero that reads as "the satellite is at the station".
    #[test]
    fn the_measurement_bindings_are_their_crate_functions() {
        let station = SlVec3 {
            x: 4_194_303.0,
            y: 172_460.0,
            z: 4_780_100.0,
        };
        let (lat, lon) = (48.8_f64.to_radians(), 2.35_f64.to_radians());
        let sat = leo();
        // Written out rather than `to_six(sat)`: routing the reference through the helper under
        // test moves both sides of every assertion together, and three mutants of `to_six` lived
        // through this test because of it.
        let six = [
            sat.position.x,
            sat.position.y,
            sat.position.z,
            sat.velocity.x,
            sat.velocity.y,
            sat.velocity.z,
        ];
        assert_eq!(to_six(sat), six);
        let fixed = Station::fixed(to_three(station));
        let topo = Topocentric::from_geodetic(to_three(station), lat, lon);

        // SAFETY: every pointer is to a local of the right type.
        unsafe {
            assert_eq!(
                sl_range(&raw const sat, &raw const station),
                m_range(&six, &fixed)
            );
            assert_eq!(
                sl_range_rate(&raw const sat, &raw const station),
                m_range_rate(&six, &fixed)
            );
            assert_eq!(
                sl_elevation(&raw const sat, &raw const station, lat, lon),
                m_elevation(&six, &topo)
            );
            assert_eq!(
                sl_azimuth(&raw const sat, &raw const station, lat, lon),
                m_azimuth(&six, &topo)
            );
        }

        // SAFETY: a null pointer is exactly the input these are documented to accept.
        unsafe {
            assert!(sl_range(core::ptr::null(), &raw const station).is_nan());
            assert!(sl_range_rate(&raw const sat, core::ptr::null()).is_nan());
            assert!(sl_elevation(core::ptr::null(), &raw const station, lat, lon).is_nan());
            assert!(sl_azimuth(&raw const sat, core::ptr::null(), lat, lon).is_nan());
        }

        // Radians, again: this station sees the satellite between the horizon and the zenith, and
        // a degrees reading of `lat` would put the station's local vertical 48.8 radians round the
        // ellipsoid and the elevation somewhere else entirely.
        // SAFETY: every pointer is to a local of the right type.
        let elevation = unsafe { sl_elevation(&raw const sat, &raw const station, lat, lon) };
        assert!(
            (-core::f64::consts::FRAC_PI_2..=core::f64::consts::FRAC_PI_2).contains(&elevation),
            "elevation {elevation} rad is not an elevation",
        );
    }

    /// Footprint geometry and orbit design, each against its crate function, plus the two failures
    /// that a caller can actually provoke: a repeat cycle of zero and an orbit too high to be
    /// Sun-synchronous.
    #[test]
    fn the_geometry_and_design_bindings_are_their_crate_functions() {
        assert_eq!(
            sl_horizon_look_angle(RE_EARTH, 500e3),
            ev_horizon_look_angle(RE_EARTH, 500e3)
        );
        assert_eq!(
            sl_critical_inclination(),
            design_critical_inclination().radians()
        );

        let (lat, lon, arc, az) = (0.4, -1.2, 0.05, 1.9);
        let got = sl_ground_point_at(
            SlLatLon {
                latitude_rad: lat,
                longitude_rad: lon,
            },
            arc,
            az,
        );
        let (want_lat, want_lon) = ev_ground_point_at(lat, lon, arc, az);
        assert_eq!((got.latitude_rad, got.longitude_rad), (want_lat, want_lon));

        let (a, inc) = (7_200e3, 1.7);
        assert_eq!(
            sl_frozen_eccentricity(a, inc, RE_EARTH, J2_EARTH, J3_EARTH),
            design_frozen_eccentricity(a, Angle::from_radians(inc), RE_EARTH, J2_EARTH, J3_EARTH),
        );

        let mut sma = 0.0;
        // SAFETY: the pointer is to a local `f64`.
        let status = unsafe {
            sl_repeat_ground_track_sma(
                43,
                3,
                inc,
                0.0,
                MU_EARTH,
                RE_EARTH,
                J2_EARTH,
                spacelab_frames::EARTH_ROTATION_RATE,
                &raw mut sma,
            )
        };
        assert_eq!(status, SL_OK);
        assert_eq!(
            sma,
            design_repeat_ground_track_sma(
                43,
                3,
                0.0,
                Angle::from_radians(inc),
                MU_EARTH,
                RE_EARTH,
                J2_EARTH,
                spacelab_frames::EARTH_ROTATION_RATE,
            ),
        );

        // Zero revolutions is not a repeat cycle. Left to the arithmetic it is a division by zero,
        // and the infinity that comes back reads as a semi-major axis until something subtracts it.
        for (revs, days) in [(0, 3), (43, 0)] {
            // SAFETY: the pointer is to a local `f64`.
            let status = unsafe {
                sl_repeat_ground_track_sma(
                    revs,
                    days,
                    inc,
                    0.0,
                    MU_EARTH,
                    RE_EARTH,
                    J2_EARTH,
                    spacelab_frames::EARTH_ROTATION_RATE,
                    &raw mut sma,
                )
            };
            assert_eq!(status, SL_ERR_INVALID_ARGUMENT, "revs {revs}, days {days}");
            assert!(sma.is_finite(), "the out-parameter must be left as it was");
        }

        let mut inclination = 0.0;
        // SAFETY: the pointer is to a local `f64`.
        let status = unsafe {
            sl_sun_synchronous_inclination(
                7_078e3,
                0.0,
                MU_EARTH,
                RE_EARTH,
                J2_EARTH,
                &raw mut inclination,
            )
        };
        assert_eq!(status, SL_OK);
        assert_eq!(
            inclination,
            design_sun_synchronous_inclination(
                7_078e3,
                0.0,
                MU_EARTH,
                RE_EARTH,
                J2_EARTH,
                SUN_SYNCHRONOUS_NODE_RATE,
            )
            .expect("a 700 km orbit is Sun-synchronous")
            .radians(),
        );
        // ~98.2 degrees, and retrograde: a Sun-synchronous orbit has to be, because J2 precesses a
        // prograde node the wrong way.
        assert!(
            inclination > core::f64::consts::FRAC_PI_2,
            "{inclination} rad is not retrograde"
        );

        // Far enough out that J2 cannot turn the node once a year, whatever the inclination.
        // SAFETY: the pointer is to a local `f64`.
        let status = unsafe {
            sl_sun_synchronous_inclination(
                42_164e3,
                0.0,
                MU_EARTH,
                RE_EARTH,
                J2_EARTH,
                &raw mut inclination,
            )
        };
        assert_eq!(status, SL_ERR_DOMAIN);
        assert!(last_error().contains("turn a year"), "{}", last_error());
    }

    /// The two-call protocol, on both generators: sizing with a null buffer, refusing a short one
    /// while still reporting the length, and filling an exact one. And the states are the crate's,
    /// not a reordering of them -- plane-major order is part of the contract.
    #[test]
    fn the_constellation_bindings_deliver_through_the_two_call_protocol() {
        let mut p = SlWalkerParams {
            pattern: WALKER_DELTA,
            inclination_rad: 0.0,
            planes: 0,
            per_plane: 0,
            phasing: 0,
            semi_major_axis_m: 0.0,
            mu: 0.0,
        };
        // SAFETY: the pointer is to a local of the right type.
        unsafe { sl_walker_params_init(&raw mut p) };
        assert_eq!(
            p.mu, MU_EARTH,
            "init must supply Earth's mu, not a zero that divides"
        );
        assert_eq!(p.pattern, WALKER_DELTA);

        p.inclination_rad = 56.0_f64.to_radians();
        p.planes = 6;
        p.per_plane = 4;
        p.phasing = 1;
        p.semi_major_axis_m = 29_600e3;

        let mut len = 0_usize;
        // SAFETY: params and length are locals; a null buffer is the documented sizing call.
        let status = unsafe { sl_walker(&raw const p, core::ptr::null_mut(), 0, &raw mut len) };
        assert_eq!(status, SL_OK);
        assert_eq!(len, 24, "6 planes of 4");

        let mut short = vec![state([0.0; 3], [0.0; 3]); 23];
        let mut short_len = 0_usize;
        // SAFETY: a buffer of 23 states described honestly as 23.
        let status = unsafe {
            sl_walker(
                &raw const p,
                short.as_mut_ptr(),
                short.len(),
                &raw mut short_len,
            )
        };
        assert_eq!(status, SL_ERR_BUFFER_TOO_SMALL);
        assert_eq!(
            short_len, 24,
            "a short buffer must still learn the length it needs"
        );
        assert_eq!(
            short[0],
            state([0.0; 3], [0.0; 3]),
            "and must not be written to"
        );

        let mut exact = vec![state([0.0; 3], [0.0; 3]); len];
        let mut exact_len = 0_usize;
        // SAFETY: a buffer of `len` states described honestly as `len`.
        let status = unsafe {
            sl_walker(
                &raw const p,
                exact.as_mut_ptr(),
                exact.len(),
                &raw mut exact_len,
            )
        };
        assert_eq!(status, SL_OK);
        assert_eq!(exact_len, len);

        let mut want = Vec::new();
        Walker {
            pattern: WalkerPattern::Delta,
            inclination: Angle::from_radians(p.inclination_rad),
            planes: p.planes,
            per_plane: p.per_plane,
            phasing: p.phasing,
            semi_major_axis: p.semi_major_axis_m,
            mu: p.mu,
        }
        .generate(|e| {
            let (r, v) = e.to_state();
            want.push(from_state(r, v));
        });
        assert_eq!(
            exact, want,
            "the binding must preserve the crate's plane-major order"
        );

        p.pattern = 9;
        // SAFETY: params and length are locals.
        let status = unsafe { sl_walker(&raw const p, core::ptr::null_mut(), 0, &raw mut len) };
        assert_eq!(
            status, SL_ERR_INVALID_ARGUMENT,
            "an unknown pattern tag must not default"
        );
        assert!(last_error().contains("pattern tag 9"), "{}", last_error());
    }

    /// Street of Coverage: the geometry solves, the generator agrees with it on the count, and an
    /// infeasible design is a named domain failure -- where the Python binding returns an empty
    /// list and leaves the caller to work out which of four reasons applied.
    #[test]
    fn the_street_of_coverage_binding_names_the_geometry_it_cannot_tile() {
        let mut p = SlSocParams {
            inclination_rad: 0.0,
            altitude_m: 0.0,
            min_elevation_rad: 0.0,
            earth_radius_m: 0.0,
            mu: 0.0,
            per_plane: 0,
        };
        // SAFETY: the pointer is to a local of the right type.
        unsafe { sl_soc_params_init(&raw mut p) };
        assert_eq!(
            (p.mu, p.earth_radius_m, p.per_plane),
            (MU_EARTH, RE_EARTH, 0)
        );

        p.inclination_rad = 90.0_f64.to_radians();
        p.altitude_m = 1_200e3;
        p.min_elevation_rad = 10.0_f64.to_radians();

        let mut g = SlSocGeometry {
            planes: 0,
            per_plane: 0,
            footprint_half_angle_rad: 0.0,
            street_half_width_rad: 0.0,
            raan_step_co_rad: 0.0,
            raan_step_counter_rad: 0.0,
            raan_step_rad: 0.0,
            raan_span_rad: 0.0,
            inter_plane_phase_rad: 0.0,
        };
        // SAFETY: both pointers are to locals of the right type.
        assert_eq!(
            unsafe { sl_street_of_coverage_geometry(&raw const p, &raw mut g) },
            SL_OK
        );
        assert!(g.planes > 0 && g.per_plane > 0, "{g:?}");
        assert!(
            g.street_half_width_rad <= g.footprint_half_angle_rad,
            "a street cannot be wider than the footprint that sweeps it: {} against {}",
            g.street_half_width_rad,
            g.footprint_half_angle_rad,
        );

        let mut len = 0_usize;
        // SAFETY: params and length are locals; a null buffer is the documented sizing call.
        let status =
            unsafe { sl_street_of_coverage(&raw const p, core::ptr::null_mut(), 0, &raw mut len) };
        assert_eq!(status, SL_OK);
        assert_eq!(
            len,
            (g.planes as usize) * (g.per_plane as usize),
            "the generator and the geometry must agree on the constellation's size",
        );

        // A footprint that reaches nowhere: the minimum elevation is the zenith, so no satellite
        // covers any ground and no number of them tiles a street.
        p.min_elevation_rad = core::f64::consts::FRAC_PI_2;
        // SAFETY: both pointers are to locals of the right type.
        assert_eq!(
            unsafe { sl_street_of_coverage_geometry(&raw const p, &raw mut g) },
            SL_ERR_DOMAIN,
        );
        // SAFETY: params and length are locals.
        let status =
            unsafe { sl_street_of_coverage(&raw const p, core::ptr::null_mut(), 0, &raw mut len) };
        assert_eq!(
            status, SL_ERR_DOMAIN,
            "the generator must fail where the geometry does"
        );
        assert!(last_error().contains("no street tiles"), "{}", last_error());
    }

    /// Every fallible entry point refuses a null out-parameter instead of writing through it. This
    /// is the one bug class where a miss is a segfault in the caller's process rather than a wrong
    /// number, so it is checked exhaustively rather than by sampling.
    #[test]
    fn every_fallible_entry_point_refuses_a_null_out_pointer() {
        let s = leo();
        let g = SlGeodetic {
            latitude_rad: 0.1,
            longitude_rad: 0.2,
            height_m: 3.0,
        };
        let mut params = SlWalkerParams {
            pattern: WALKER_DELTA,
            inclination_rad: 0.9,
            planes: 2,
            per_plane: 2,
            phasing: 0,
            semi_major_axis_m: 7_000e3,
            mu: MU_EARTH,
        };
        let mut soc = SlSocParams {
            inclination_rad: core::f64::consts::FRAC_PI_2,
            altitude_m: 1_200e3,
            min_elevation_rad: 0.17,
            earth_radius_m: RE_EARTH,
            mu: MU_EARTH,
            per_plane: 0,
        };
        let mut len = 0_usize;
        let mut sink = s;

        // SAFETY: every call passes a null where the function documents an out-parameter, which is
        // precisely the input each one is required to reject before dereferencing anything.
        unsafe {
            assert_eq!(
                sl_teme_to_itrf(&raw const s, 0, 0, 0.0, 0.0, 0.0, core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_gcrf_to_itrf(
                    &raw const s,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_itrf_to_geodetic(&raw const s.position, core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_geodetic_to_itrf(&raw const g, core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_sgp4_open(c"a".as_ptr(), c"b".as_ptr(), core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_sgp4_propagate_at(core::ptr::null(), 0.0, core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_sgp4_propagate_batch(
                    core::ptr::null(),
                    core::ptr::null(),
                    0,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_kepler_propagate(&raw const s, MU_EARTH, 1.0, core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_propagate_two_body(
                    &raw const s,
                    MU_EARTH,
                    1.0,
                    1,
                    PRECISION_DOUBLE,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_lambert(
                    &raw const s.position,
                    &raw const s.position,
                    1.0,
                    MU_EARTH,
                    true,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                ),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_repeat_ground_track_sma(
                    43,
                    3,
                    1.0,
                    0.0,
                    MU_EARTH,
                    RE_EARTH,
                    J2_EARTH,
                    spacelab_frames::EARTH_ROTATION_RATE,
                    core::ptr::null_mut(),
                ),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_sun_synchronous_inclination(
                    7_078e3,
                    0.0,
                    MU_EARTH,
                    RE_EARTH,
                    J2_EARTH,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_walker(
                    &raw const params,
                    core::ptr::null_mut(),
                    0,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_street_of_coverage_geometry(&raw const soc, core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_street_of_coverage(
                    &raw const soc,
                    core::ptr::null_mut(),
                    0,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT,
            );

            // And a null *input* where one is required, which must be refused just as firmly.
            assert_eq!(
                sl_walker(core::ptr::null(), core::ptr::null_mut(), 0, &raw mut len),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_street_of_coverage(core::ptr::null(), core::ptr::null_mut(), 0, &raw mut len),
                SL_ERR_INVALID_ARGUMENT,
            );
            assert_eq!(
                sl_kepler_propagate(core::ptr::null(), MU_EARTH, 1.0, &raw mut sink),
                SL_ERR_INVALID_ARGUMENT,
            );

            // The `_init` helpers take a null and do nothing, so cleanup code needs no branch.
            sl_walker_params_init(core::ptr::null_mut());
            sl_soc_params_init(core::ptr::null_mut());

            // And on a real pointer they overwrite it, which is also what reads these two back --
            // the compiler cannot see a write through a raw pointer, so without a read afterwards
            // the initialisers above look like dead stores.
            sl_walker_params_init(&raw mut params);
            sl_soc_params_init(&raw mut soc);
        }
        assert_eq!(params.planes, 0);
        assert_eq!(soc.per_plane, 0);
    }

    /// A caller asking only for the length gets it, and gets nothing written -- through a null
    /// destination, and through a real one whose capacity leaves no room even for the terminator.
    #[test]
    fn sizing_a_buffer_writes_to_none_of_it() {
        let text = "the length is the answer";
        let needed = text.len() + 1;

        // The documented sizing call. Nothing is written, because there is nowhere to write.
        assert_eq!(copy_to_c_buffer(text, core::ptr::null_mut(), 0), needed);

        // A null destination with a capacity is still a null destination -- a caller that reports
        // room it does not own gets the length back, not a write through null.
        assert_eq!(copy_to_c_buffer(text, core::ptr::null_mut(), 64), needed);

        // A real buffer of no capacity: the length, and the buffer untouched. There is no room even
        // for the terminator, so writing one would overrun by a byte.
        let mut guarded = [0x5a_u8; 4];
        assert_eq!(
            copy_to_c_buffer(text, guarded.as_mut_ptr().cast::<c_char>(), 0),
            needed
        );
        assert_eq!(guarded, [0x5a; 4]);
    }

    /// The message buffer sizes itself, terminates whatever it writes, and truncates on a UTF-8
    /// boundary. The last part is why this is a test and not an assumption: cutting mid-sequence
    /// leaves the caller holding bytes that are a valid C string and an invalid `std::string`.
    #[test]
    fn the_error_message_sizes_terminates_and_truncates_on_a_character_boundary() {
        // A message with a multi-byte character at a place a naive truncation would split.
        fail(SL_ERR_DOMAIN, "ångström");

        // SAFETY: the documented sizing call.
        let needed = unsafe { sl_last_error_message(core::ptr::null_mut(), 0) };
        assert_eq!(needed, "ångström".len() + 1);

        for cap in 1..=needed {
            let mut buf = vec![0x7f_u8; needed + 4];
            // SAFETY: `buf` is writable for far more than `cap` bytes.
            let again = unsafe { sl_last_error_message(buf.as_mut_ptr().cast::<c_char>(), cap) };
            assert_eq!(
                again, needed,
                "the needed length must not depend on the buffer given"
            );

            let nul = buf.iter().position(|&b| b == 0).expect("always terminated");
            assert!(nul < cap, "the terminator must land inside the buffer");
            let written = core::str::from_utf8(&buf[..nul])
                .unwrap_or_else(|e| panic!("cap {cap} truncated mid-character: {e}"));
            assert!(
                "ångström".starts_with(written),
                "cap {cap} wrote {written:?}"
            );
            assert!(
                buf[cap..].iter().all(|&b| b == 0x7f),
                "cap {cap} wrote past the buffer it was given",
            );
        }

        // Nothing has failed on a fresh thread, so it reports the empty string -- not the message
        // this thread just set, which is the whole point of it being thread-local.
        let other = std::thread::spawn(|| {
            // SAFETY: the documented sizing call, on a thread that has not failed.
            let needed = unsafe { sl_last_error_message(core::ptr::null_mut(), 0) };
            assert_eq!(needed, 1, "a fresh thread has no error to report");
        });
        other.join().expect("the thread does not panic");
    }

    /// A panic inside an entry point is contained and named, rather than unwinding into a caller
    /// that has no idea what a Rust panic is. Provoked through `guard` directly, because no input
    /// to the public surface is supposed to reach one.
    #[test]
    fn a_panic_at_the_boundary_is_contained_and_named() {
        let status = guard(|| panic!("the force model came apart"));
        assert_eq!(status, SL_ERR_PANIC);
        assert!(
            last_error().contains("the force model came apart"),
            "{}",
            last_error()
        );

        let value = guard_f64(|| panic!("and so did this one"));
        assert!(
            value.is_nan(),
            "a total function reports a contained panic as NaN"
        );
        assert!(
            last_error().contains("and so did this one"),
            "{}",
            last_error()
        );
    }

    /// Earth's gravitational parameter, as the header states it.
    const SL_MU_EARTH_TEST: f64 = 3.986_004_418e14;

    /// The IAS15 ephemeris is the crate's own integration, and the interpolant is the orbit: with
    /// J2 switched off it must agree with the two-body Kepler propagator, which is an independent
    /// implementation of the same physics rather than a second call to the same code.
    #[test]
    fn the_j2_ephemeris_is_the_integration_and_its_interpolant_is_the_orbit() {
        let start = leo();
        let span = 5_400.0;
        let mut handle: *mut SlEphemeris = core::ptr::null_mut();
        // SAFETY: a local state and a local out pointer.
        assert_eq!(
            unsafe {
                sl_ephemeris_j2(
                    &raw const start,
                    span,
                    SL_MU_EARTH_TEST,
                    0.0,
                    6_378_137.0,
                    &raw mut handle,
                )
            },
            SL_OK
        );
        assert!(!handle.is_null());

        let (mut t0, mut t1) = (f64::NAN, f64::NAN);
        // SAFETY: a live handle and two local out pointers.
        assert_eq!(
            unsafe { sl_ephemeris_span(handle, &raw mut t0, &raw mut t1) },
            SL_OK
        );
        assert_eq!((t0, t1), (0.0, span));

        let (mut accepted, mut rejected) = (0_u32, 0_u32);
        // SAFETY: a live handle and two local out pointers.
        assert_eq!(
            unsafe { sl_ephemeris_step_counts(handle, &raw mut accepted, &raw mut rejected) },
            SL_OK
        );
        assert!(
            accepted > 1 && rejected < accepted,
            "an arc of {span} s is many steps, not the {accepted} accepted and {rejected} rejected \
             a constant would report"
        );

        for &t in &[0.0, 900.0, 2_700.0, span] {
            let mut got = state([0.0; 3], [0.0; 3]);
            // SAFETY: a live handle and a local out pointer.
            assert_eq!(unsafe { sl_ephemeris_at(handle, t, &raw mut got) }, SL_OK);
            let (r, v) = astro_kepler(
                to_vec3(start.position),
                to_vec3(start.velocity),
                SL_MU_EARTH_TEST,
                t,
            );
            let want = from_state(r, v);
            let dr = ((got.position.x - want.position.x).powi(2)
                + (got.position.y - want.position.y).powi(2)
                + (got.position.z - want.position.z).powi(2))
            .sqrt();
            assert!(
                dr < 1e-3,
                "with J2 off the interpolant is the two-body orbit; at t = {t} it is {dr} m away"
            );
        }

        // The batch is the singles, exactly, and the loop is only there to save the crossings.
        let times = [0.0, 900.0, 2_700.0, span];
        let mut batch = [state([0.0; 3], [0.0; 3]); 4];
        // SAFETY: a live handle, a local array of four epochs and a local array of four states.
        assert_eq!(
            unsafe { sl_ephemeris_sample(handle, times.as_ptr(), times.len(), batch.as_mut_ptr()) },
            SL_OK
        );
        for (i, &t) in times.iter().enumerate() {
            let mut one = state([0.0; 3], [0.0; 3]);
            // SAFETY: a live handle and a local out pointer.
            assert_eq!(unsafe { sl_ephemeris_at(handle, t, &raw mut one) }, SL_OK);
            assert_eq!(batch[i], one, "batch and single disagree at t = {t}");
        }

        // SAFETY: a live handle, closed exactly once.
        unsafe { sl_ephemeris_close(handle) };
        // SAFETY: the header documents a null close as a no-op.
        unsafe { sl_ephemeris_close(core::ptr::null_mut()) };
    }

    /// Off the span there is no answer, and an interpolant that extrapolated would give a wrong
    /// one that looks right. Checked at both ends and for the whole batch, which must refuse
    /// without writing anything.
    #[test]
    fn an_ephemeris_refuses_to_extrapolate() {
        let start = leo();
        let mut handle: *mut SlEphemeris = core::ptr::null_mut();
        // SAFETY: a local state and a local out pointer.
        assert_eq!(
            unsafe {
                sl_ephemeris_j2(
                    &raw const start,
                    600.0,
                    SL_MU_EARTH_TEST,
                    J2_EARTH,
                    6_378_137.0,
                    &raw mut handle,
                )
            },
            SL_OK
        );

        let mut got = state([1.0; 3], [2.0; 3]);
        let sentinel = got;
        for &t in &[-1.0, 600.001, f64::NAN] {
            // SAFETY: a live handle and a local out pointer.
            assert_eq!(
                unsafe { sl_ephemeris_at(handle, t, &raw mut got) },
                SL_ERR_DOMAIN,
                "t = {t} is off the span"
            );
            assert_eq!(got, sentinel, "a refused call must not write");
        }

        let times = [0.0, 300.0, 900.0];
        let mut batch = [sentinel; 3];
        // SAFETY: a live handle and local arrays of three.
        assert_eq!(
            unsafe { sl_ephemeris_sample(handle, times.as_ptr(), times.len(), batch.as_mut_ptr()) },
            SL_ERR_DOMAIN,
            "one bad epoch refuses the whole batch"
        );
        assert_eq!(
            batch, [sentinel; 3],
            "a refused batch must not be partly written"
        );

        // SAFETY: a live handle, closed exactly once.
        unsafe { sl_ephemeris_close(handle) };
    }

    /// The geopotential ephemeris is the embedded model, and a degree past it is a domain error
    /// rather than a silent truncation to whatever the table happens to hold.
    #[test]
    fn the_geopotential_ephemeris_is_the_embedded_model() {
        let start = leo();
        let mut handle: *mut SlEphemeris = core::ptr::null_mut();
        // SAFETY: a local state and a local out pointer.
        assert_eq!(
            unsafe {
                sl_ephemeris_geopotential(
                    &raw const start,
                    600.0,
                    MODEL_MAX_DEGREE as u32 + 1,
                    0.0,
                    1e-9,
                    &raw mut handle,
                )
            },
            SL_ERR_DOMAIN
        );
        assert!(
            handle.is_null(),
            "a refused call must not hand back a handle"
        );

        // SAFETY: a local state and a local out pointer.
        assert_eq!(
            unsafe {
                sl_ephemeris_geopotential(&raw const start, 600.0, 8, 0.3, 1e-9, &raw mut handle)
            },
            SL_OK
        );
        assert!(!handle.is_null());

        let mut got = state([0.0; 3], [0.0; 3]);
        // SAFETY: a live handle and a local out pointer.
        assert_eq!(
            unsafe { sl_ephemeris_at(handle, 600.0, &raw mut got) },
            SL_OK
        );

        // A degree-8 field is not a point mass, and the difference over ten minutes is the whole
        // reason this entry point exists: pin it as a range rather than a value, because the value
        // is the model's and this test is about the binding.
        let (r, _) = astro_kepler(
            to_vec3(start.position),
            to_vec3(start.velocity),
            SL_MU_EARTH_TEST,
            600.0,
        );
        let dr = ((got.position.x - r.x).powi(2)
            + (got.position.y - r.y).powi(2)
            + (got.position.z - r.z).powi(2))
        .sqrt();
        assert!(
            (100.0..100_000.0).contains(&dr),
            "a degree-8 field should move a LEO by hundreds of metres in ten minutes, not {dr} m"
        );

        // SAFETY: a live handle, closed exactly once.
        unsafe { sl_ephemeris_close(handle) };
    }

    /// A zero span is not an arc, and the two constructors say so rather than integrating nothing.
    #[test]
    fn the_ephemeris_constructors_refuse_an_empty_span() {
        let start = leo();
        let mut handle: *mut SlEphemeris = core::ptr::null_mut();
        // SAFETY: a local state and a local out pointer.
        assert_eq!(
            unsafe {
                sl_ephemeris_j2(
                    &raw const start,
                    0.0,
                    SL_MU_EARTH_TEST,
                    0.0,
                    6_378_137.0,
                    &raw mut handle,
                )
            },
            SL_ERR_INVALID_ARGUMENT
        );
        // SAFETY: a local state and a local out pointer.
        assert_eq!(
            unsafe {
                sl_ephemeris_geopotential(&raw const start, f64::NAN, 4, 0.0, 1e-9, &raw mut handle)
            },
            SL_ERR_INVALID_ARGUMENT
        );
        // SAFETY: a local state and a local out pointer.
        assert_eq!(
            unsafe {
                sl_ephemeris_geopotential(&raw const start, 600.0, 4, 0.0, 0.0, &raw mut handle)
            },
            SL_ERR_INVALID_ARGUMENT,
            "a zero tolerance is not a tolerance"
        );
        assert!(handle.is_null());
    }

    /// The acceleration binding is the crate's own, bit for bit.
    #[test]
    fn the_geopotential_acceleration_binding_is_its_crate_function() {
        let r = SlVec3 {
            x: 6_878_137.0,
            y: 1_234_567.0,
            z: -2_345_678.0,
        };
        let mut got = SlVec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        // SAFETY: two local values.
        assert_eq!(
            unsafe { sl_geopotential_acceleration(12, &raw const r, &raw mut got) },
            SL_OK
        );
        let want = GeoPotential::embedded(12)
            .expect("degree 12 is inside the embedded model")
            .acceleration(0.0, &to_three(r), &[0.0_f64; 3]);
        assert_eq!([got.x, got.y, got.z], want);

        // SAFETY: two local values.
        assert_eq!(
            unsafe {
                sl_geopotential_acceleration(
                    MODEL_MAX_DEGREE as u32 + 1,
                    &raw const r,
                    &raw mut got,
                )
            },
            SL_ERR_DOMAIN
        );
    }

    /// The libration points are the crate's, in the order the header promises, and the collinear
    /// three lie on the x-axis while the triangular two are the equilateral pair — which is what
    /// would catch a binding that wrote them out in the wrong order.
    #[test]
    fn the_libration_points_are_the_five_in_the_documented_order() {
        let mu = 0.012_150_584_460_351;
        let mut got = [SlVec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }; 5];
        // SAFETY: a local array of five.
        assert_eq!(unsafe { sl_libration_points(mu, got.as_mut_ptr()) }, SL_OK);

        let want = Cr3bp { mu }.libration_points();
        for (i, p) in want.iter().enumerate() {
            assert_eq!(got[i], from_vec3(*p), "libration point {} disagrees", i + 1);
        }
        for p in got.iter().take(3) {
            assert_eq!(p.y, 0.0, "L1..L3 are collinear");
        }
        assert!(got[3].y > 0.0 && got[4].y < 0.0, "L4 leads and L5 trails");

        // SAFETY: a local array of five.
        assert_eq!(
            unsafe { sl_libration_points(1.5, got.as_mut_ptr()) },
            SL_ERR_DOMAIN,
            "a mass ratio outside (0, 1) is not a system"
        );
        // The interval is open at both ends, and both ends are asked for: at 0 and at 1 one of the
        // primaries has no mass, which is a two-body problem wearing a three-body interface.
        for degenerate in [0.0, 1.0] {
            // SAFETY: a local array of five.
            assert_eq!(
                unsafe { sl_libration_points(degenerate, got.as_mut_ptr()) },
                SL_ERR_DOMAIN,
                "a mass ratio of {degenerate} is not a system"
            );
        }
    }

    /// Richardson's guess is the crate's, and an impossible libration point is a domain error.
    #[test]
    fn the_richardson_halo_binding_is_its_crate_function() {
        let mu = 0.012_150_584_460_351;
        let mut got = state([0.0; 3], [0.0; 3]);
        // SAFETY: a local out pointer.
        assert_eq!(
            unsafe { sl_richardson_halo(1, 0.002, true, mu, &raw mut got) },
            SL_OK
        );
        let want = richardson_halo_guess(Cr3bp { mu }, 0, 0.002, true).expect("L1 has a guess");
        assert_eq!(
            got,
            from_six(want),
            "the C API numbers from 1 where the crate indexes from 0"
        );

        // L2 is the other one the third-order theory covers; L3 and 0 are not points it can answer
        // for, and each is a domain error rather than an off-by-one into the wrong orbit.
        // SAFETY: a local out pointer.
        assert_eq!(
            unsafe { sl_richardson_halo(2, 0.002, true, mu, &raw mut got) },
            SL_OK
        );
        assert_eq!(
            got,
            from_six(richardson_halo_guess(Cr3bp { mu }, 1, 0.002, true).expect("L2 has a guess"))
        );
        for li in [0, 3, 9] {
            // SAFETY: a local out pointer.
            assert_eq!(
                unsafe { sl_richardson_halo(li, 0.002, true, mu, &raw mut got) },
                SL_ERR_DOMAIN,
                "L{li} has no third-order guess"
            );
        }
    }

    /// Every new fallible entry point refuses a null out-pointer rather than writing through it.
    #[test]
    fn the_propagation_entry_points_refuse_a_null_out_pointer() {
        let start = leo();
        let r = SlVec3 {
            x: 7e6,
            y: 0.0,
            z: 0.0,
        };
        // SAFETY: every call below is given a null where the header says a pointer is required,
        // which is exactly the contract under test.
        unsafe {
            assert_eq!(
                sl_ephemeris_j2(
                    &raw const start,
                    600.0,
                    SL_MU_EARTH_TEST,
                    0.0,
                    6.4e6,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_ephemeris_geopotential(
                    core::ptr::null(),
                    600.0,
                    4,
                    0.0,
                    1e-9,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_ephemeris_at(core::ptr::null(), 0.0, core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_ephemeris_sample(
                    core::ptr::null(),
                    core::ptr::null(),
                    0,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_ephemeris_span(
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_ephemeris_step_counts(
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_geopotential_acceleration(4, &raw const r, core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_libration_points(0.5, core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_richardson_halo(1, 0.002, true, 0.5, core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT
            );
        }
    }

    /// A LEO in the mean equinoctial elements DSST works in: `a` 7000 km, `e` 0.001, `i` 51.6°,
    /// node and perigee at zero. `q = tan(i/2)` because the node is zero, and `k = e` because the
    /// longitude of perigee is.
    fn leo_equinoctial() -> [f64; 6] {
        let inc: f64 = 51.6_f64.to_radians();
        [7.0e6, 1.0e-3, 0.0, (inc / 2.0).tan(), 0.0, 0.0]
    }

    /// Run one of the four DSST entry points over `times`, or return the status it refused with.
    fn dsst_run(
        entry: unsafe extern "C" fn(
            *const SlDsstParams,
            *const f64,
            *const f64,
            usize,
            *mut f64,
        ) -> i32,
        params: &SlDsstParams,
        initial: &[f64; 6],
        times: &[f64],
    ) -> Result<Vec<[f64; 6]>, i32> {
        let mut flat = vec![0.0_f64; 6 * times.len()];
        // SAFETY: `initial` is six doubles, `times` is `times.len()`, and `flat` is `6 *` that.
        let status = unsafe {
            entry(
                params,
                initial.as_ptr(),
                times.as_ptr(),
                times.len(),
                flat.as_mut_ptr(),
            )
        };
        if status != SL_OK {
            return Err(status);
        }
        Ok(flat.as_chunks::<6>().0.to_vec())
    }

    fn default_dsst() -> SlDsstParams {
        let mut p = SlDsstParams {
            mu: 0.0,
            earth_radius_m: 0.0,
            epoch_j2000_seconds: 0.0,
            reflectivity: 0.0,
            area_over_mass: 0.0,
            mu_sun: 0.0,
            mu_moon: 0.0,
            zonal: [0.0; SL_DSST_MAX_ZONAL],
            zonal_count: 0,
        };
        // SAFETY: `p` is one initialised `SlDsstParams`.
        unsafe { sl_dsst_params_init(&raw mut p) };
        p
    }

    /// The whole point of a mean-element theory: the zonal field does not change the mean axis, so
    /// whatever else drifts, `a` does not. The node does — westward for a prograde orbit — and the
    /// short-period corrections put back the oscillation the averaging removed.
    #[test]
    fn the_mean_elements_conserve_the_axis_and_the_osculating_ones_do_not() {
        let p = default_dsst();
        assert_eq!(p.zonal_count, 4);
        let y0 = leo_equinoctial();
        let times = [0.0, 1800.0, 5400.0];

        let mean = dsst_run(sl_dsst_zonal_mean, &p, &y0, &times).expect("a LEO propagates");
        for state in &mean {
            assert!(
                (state[0] - y0[0]).abs() / y0[0] < 1e-12,
                "the zonal mean rate leaves a alone, got {}",
                state[0]
            );
        }
        assert_eq!(mean[0], y0, "t = 0 is the initial state itself");

        // λ = M + ϖ advances at about the mean motion, √(μ/a³) ≈ 1.078e-3 rad/s.
        let n = (p.mu / y0[0].powi(3)).sqrt();
        let advanced = mean[1][5] - mean[0][5];
        assert!(
            (advanced - n * 1800.0).abs() < 0.05 * n * 1800.0,
            "λ advanced {advanced}, expected about {}",
            n * 1800.0
        );

        // The node regresses for a prograde orbit, so `p = tan(i/2)·sin Ω` leaves zero going
        // negative. J2 is positive, i is 51.6°, and the sign of dΩ/dt is that of −cos i.
        assert!(
            mean[2][4] < 0.0,
            "the node should regress, p went to {}",
            mean[2][4]
        );

        let osculating =
            dsst_run(sl_dsst_zonal_osculating, &p, &y0, &times).expect("a LEO propagates");
        let short_period = (osculating[1][0] - mean[1][0]).abs();
        assert!(
            (1.0..50e3).contains(&short_period),
            "the J2 short-period in a is kilometres, got {short_period}"
        );
    }

    /// The combined propagator is the zonal one plus three more contributions, so it must answer
    /// differently — and it must be the same at `t = 0`, where nothing has been integrated yet.
    #[test]
    fn the_combined_propagator_adds_to_the_zonal_one() {
        let p = default_dsst();
        let y0 = leo_equinoctial();
        let times = [0.0, 86_400.0];

        let zonal = dsst_run(sl_dsst_zonal_mean, &p, &y0, &times).expect("a LEO propagates");
        let full = dsst_run(sl_dsst_mean, &p, &y0, &times).expect("a LEO propagates");
        assert_eq!(full[0], y0, "t = 0 is the initial state itself");

        let separation: f64 = zonal[1]
            .iter()
            .zip(&full[1])
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            separation > 0.0,
            "the Sun, the Moon and SRP contribute nothing over a day"
        );

        let osculating = dsst_run(sl_dsst_osculating, &p, &y0, &times).expect("a LEO propagates");
        assert_ne!(osculating[1], full[1], "the short-period terms are missing");
    }

    #[test]
    fn a_dsst_request_that_cannot_be_answered_names_what_is_wrong_with_it() {
        let y0 = leo_equinoctial();
        let times = [0.0];

        let mut p = default_dsst();
        p.zonal_count = 0;
        assert_eq!(
            dsst_run(sl_dsst_zonal_mean, &p, &y0, &times),
            Err(SL_ERR_INVALID_ARGUMENT)
        );
        assert!(last_error().contains("zonal_count is 0"));

        p.zonal_count = SL_DSST_MAX_ZONAL as u32 + 1;
        assert_eq!(
            dsst_run(sl_dsst_zonal_mean, &p, &y0, &times),
            Err(SL_ERR_INVALID_ARGUMENT)
        );

        // The bound is inclusive, and the only way to say so is to fill it.
        p.zonal_count = SL_DSST_MAX_ZONAL as u32;
        p.zonal = [1e-6; SL_DSST_MAX_ZONAL];
        p.zonal[0] = 1.082_626_68e-3;
        assert!(
            dsst_run(sl_dsst_zonal_mean, &p, &y0, &times).is_ok(),
            "a full zonal set is the largest one accepted, not the first one refused: {}",
            last_error()
        );

        // The radius is checked the same way `mu` is -- finite and positive, both halves.
        for bad in [0.0, f64::INFINITY, f64::NAN] {
            p = default_dsst();
            p.earth_radius_m = bad;
            assert_eq!(
                dsst_run(sl_dsst_zonal_mean, &p, &y0, &times),
                Err(SL_ERR_INVALID_ARGUMENT),
                "an earth_radius_m of {bad} is not a radius"
            );
        }

        p = default_dsst();
        p.zonal[2] = f64::NAN;
        assert_eq!(
            dsst_run(sl_dsst_zonal_mean, &p, &y0, &times),
            Err(SL_ERR_INVALID_ARGUMENT)
        );
        assert!(last_error().contains("J4 is not finite"));

        p = default_dsst();
        p.mu = 0.0;
        assert_eq!(
            dsst_run(sl_dsst_mean, &p, &y0, &times),
            Err(SL_ERR_INVALID_ARGUMENT)
        );

        p = default_dsst();
        p.area_over_mass = f64::INFINITY;
        assert_eq!(
            dsst_run(sl_dsst_mean, &p, &y0, &times),
            Err(SL_ERR_INVALID_ARGUMENT)
        );
        // The zonal entry points never read it, so the same parameters are fine there.
        assert!(dsst_run(sl_dsst_zonal_mean, &p, &y0, &times).is_ok());

        let p = default_dsst();
        let mut collapsed = y0;
        collapsed[0] = 0.0;
        assert_eq!(
            dsst_run(sl_dsst_zonal_mean, &p, &collapsed, &times),
            Err(SL_ERR_DOMAIN)
        );

        assert_eq!(
            dsst_run(sl_dsst_zonal_mean, &p, &y0, &[0.0, f64::NAN]),
            Err(SL_ERR_INVALID_ARGUMENT)
        );
        assert_eq!(
            dsst_run(sl_dsst_zonal_mean, &p, &y0, &[]),
            Err(SL_ERR_INVALID_ARGUMENT)
        );

        // SAFETY: every pointer but the one under test is valid.
        assert_eq!(
            unsafe {
                sl_dsst_zonal_mean(
                    core::ptr::null(),
                    y0.as_ptr(),
                    times.as_ptr(),
                    1,
                    [0.0_f64; 6].as_mut_ptr(),
                )
            },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    /// The model the Gauss-Newton tests solve: six independent measurements of the six components,
    /// so `residual = x − target` with an identity Jacobian. Its solution is the target exactly and
    /// its covariance is `diag(1/weight)`, both known without running anything.
    struct Diagonal {
        target: [f64; 6],
        weight: f64,
        calls: core::cell::Cell<u32>,
    }

    unsafe extern "C" fn diagonal_rows(
        user_data: *mut c_void,
        state: *const f64,
        rows: *mut SlResidualRow,
        row_count: usize,
    ) -> i32 {
        // SAFETY: the library hands back the `user_data` it was given, and the two buffers it
        // documents; this callback is only ever installed with a `Diagonal` behind it.
        let model = unsafe { &*user_data.cast::<Diagonal>() };
        model.calls.set(model.calls.get() + 1);
        // SAFETY: as above.
        let (x, rows) = unsafe {
            (
                core::slice::from_raw_parts(state, 6),
                core::slice::from_raw_parts_mut(rows, row_count),
            )
        };
        for (i, row) in rows.iter_mut().enumerate() {
            let mut jacobian = [0.0; 6];
            jacobian[i] = 1.0;
            *row = SlResidualRow {
                jacobian,
                residual: x[i] - model.target[i],
                weight: model.weight,
            };
        }
        SL_OK
    }

    unsafe extern "C" fn refusing_rows(
        _user_data: *mut c_void,
        _state: *const f64,
        _rows: *mut SlResidualRow,
        _row_count: usize,
    ) -> i32 {
        SL_ERR_PARSE
    }

    fn solve_diagonal(model: &Diagonal, x0: [f64; 6]) -> Result<SlEstimate, i32> {
        let mut out = SlEstimate {
            state: [0.0; 6],
            covariance: [0.0; 36],
            cost: 0.0,
            iterations: 0,
            converged: false,
        };
        // SAFETY: `x0` is six doubles, `out` is one estimate, and the callback is the one written
        // for this model.
        let status = unsafe {
            sl_gauss_newton(
                core::ptr::null(),
                x0.as_ptr(),
                Some(diagonal_rows),
                (model as *const Diagonal).cast::<c_void>().cast_mut(),
                6,
                &raw mut out,
            )
        };
        if status == SL_OK {
            Ok(out)
        } else {
            Err(status)
        }
    }

    #[test]
    fn the_least_squares_solve_recovers_a_state_whose_answer_is_known_in_closed_form() {
        let model = Diagonal {
            target: [1.0, -2.0, 3.5, 0.25, -0.75, 10.0],
            weight: 4.0,
            calls: core::cell::Cell::new(0),
        };
        let solved = solve_diagonal(&model, [0.0; 6]).expect("a full-rank diagonal system solves");

        assert!(solved.converged);
        assert!(solved.iterations >= 1);
        assert!(model.calls.get() >= solved.iterations);
        for (got, want) in solved.state.iter().zip(&model.target) {
            assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        }
        // Six independent unit rows of weight w give `(HᵀWH)⁻¹ = diag(1/w)`, and nothing else.
        for i in 0..6 {
            for j in 0..6 {
                let want = if i == j { 1.0 / model.weight } else { 0.0 };
                let got = solved.covariance[i * 6 + j];
                assert!((got - want).abs() < 1e-12, "covariance[{i}][{j}] = {got}");
            }
        }
        assert!(
            solved.cost < 1e-18,
            "an exactly solvable system has no cost"
        );
    }

    #[test]
    fn the_least_squares_options_are_honoured_and_checked() {
        let mut o = SlGaussNewtonOptions {
            max_iterations: 0,
            cost_tolerance: 0.0,
            step_tolerance: 0.0,
            initial_lambda: 0.0,
            absolute_cost_floor: 0.0,
        };
        // SAFETY: `o` is one initialised `SlGaussNewtonOptions`.
        unsafe { sl_gauss_newton_options_init(&raw mut o) };
        let d = GaussNewton::default();
        assert_eq!(o.max_iterations, d.max_iterations);
        assert_eq!(o.initial_lambda, d.initial_lambda);

        let model = Diagonal {
            target: [1.0; 6],
            weight: 1.0,
            calls: core::cell::Cell::new(0),
        };
        let mut out = SlEstimate {
            state: [0.0; 6],
            covariance: [0.0; 36],
            cost: 0.0,
            iterations: 0,
            converged: false,
        };
        let user = (&raw const model).cast::<c_void>().cast_mut();
        let x0 = [0.0_f64; 6];

        // One iteration is not a failure: it is an answer with `converged` false, and the state it
        // reached is still worth looking at.
        o.max_iterations = 1;
        // SAFETY: as in `solve_diagonal`.
        assert_eq!(
            unsafe {
                sl_gauss_newton(
                    &raw const o,
                    x0.as_ptr(),
                    Some(diagonal_rows),
                    user,
                    6,
                    &raw mut out,
                )
            },
            SL_OK
        );
        assert_eq!(out.iterations, 1);
        assert!(!out.converged);

        o.max_iterations = 0;
        // SAFETY: as above.
        assert_eq!(
            unsafe {
                sl_gauss_newton(
                    &raw const o,
                    x0.as_ptr(),
                    Some(diagonal_rows),
                    user,
                    6,
                    &raw mut out,
                )
            },
            SL_ERR_INVALID_ARGUMENT
        );

        // SAFETY: as above.
        unsafe { sl_gauss_newton_options_init(&raw mut o) };
        o.cost_tolerance = -1.0;
        // SAFETY: as above.
        assert_eq!(
            unsafe {
                sl_gauss_newton(
                    &raw const o,
                    x0.as_ptr(),
                    Some(diagonal_rows),
                    user,
                    6,
                    &raw mut out,
                )
            },
            SL_ERR_INVALID_ARGUMENT
        );
        assert!(last_error().contains("cost_tolerance"));
    }

    #[test]
    fn a_least_squares_request_that_cannot_be_answered_names_what_is_wrong_with_it() {
        let model = Diagonal {
            target: [0.0; 6],
            weight: 1.0,
            calls: core::cell::Cell::new(0),
        };
        let user = (&raw const model).cast::<c_void>().cast_mut();
        let x0 = [0.0_f64; 6];
        let mut out = SlEstimate {
            state: [0.0; 6],
            covariance: [0.0; 36],
            cost: 0.0,
            iterations: 0,
            converged: false,
        };

        // SAFETY: every pointer but the one under test is valid.
        unsafe {
            assert_eq!(
                sl_gauss_newton(core::ptr::null(), x0.as_ptr(), None, user, 6, &raw mut out),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_gauss_newton(
                    core::ptr::null(),
                    x0.as_ptr(),
                    Some(diagonal_rows),
                    user,
                    0,
                    &raw mut out
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_gauss_newton(
                    core::ptr::null(),
                    core::ptr::null(),
                    Some(diagonal_rows),
                    user,
                    6,
                    &raw mut out
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_gauss_newton(
                    core::ptr::null(),
                    [f64::NAN; 6].as_ptr(),
                    Some(diagonal_rows),
                    user,
                    6,
                    &raw mut out
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            // A model that refuses stops the solve and says so, with the code it returned in the
            // message rather than passed off as a status of ours.
            assert_eq!(
                sl_gauss_newton(
                    core::ptr::null(),
                    x0.as_ptr(),
                    Some(refusing_rows),
                    user,
                    6,
                    &raw mut out
                ),
                SL_ERR_COMPUTE
            );
        }
        assert!(last_error().contains("returning 3"), "{}", last_error());

        // Fewer independent rows than the state has directions: the system has no unique solution
        // and saying so beats returning one of the infinitely many.
        let one_row = Diagonal {
            target: [1.0; 6],
            weight: 1.0,
            calls: core::cell::Cell::new(0),
        };
        let mut narrow = SlEstimate {
            state: [0.0; 6],
            covariance: [0.0; 36],
            cost: 0.0,
            iterations: 0,
            converged: false,
        };
        // SAFETY: as above; `row_count` of 1 is what the callback is told and what it fills.
        assert_eq!(
            unsafe {
                sl_gauss_newton(
                    core::ptr::null(),
                    x0.as_ptr(),
                    Some(diagonal_rows),
                    (&raw const one_row).cast::<c_void>().cast_mut(),
                    1,
                    &raw mut narrow,
                )
            },
            SL_ERR_DOMAIN
        );
    }

    /// Open a filter and fold `rows` in, so each test says what it is testing and not how a handle
    /// is opened.
    fn srif_with(rows: &[([f64; 6], f64, f64)]) -> *mut SlSrif {
        let mut f: *mut SlSrif = core::ptr::null_mut();
        // SAFETY: `f` is one writable pointer; a null prior means an empty filter.
        let status = unsafe { sl_srif_open(core::ptr::null(), core::ptr::null(), &raw mut f) };
        assert_eq!(status, SL_OK);
        assert!(!f.is_null());
        for (h, z, w) in rows {
            // SAFETY: `f` is the handle just opened and `h` is six doubles.
            assert_eq!(unsafe { sl_srif_add_row(f, h.as_ptr(), *z, *w) }, SL_OK);
        }
        f
    }

    #[test]
    fn the_information_filter_solves_the_system_its_rows_describe() {
        let target = [1.0, -2.0, 3.5, 0.25, -0.75, 10.0];
        let weight = 4.0;
        let rows: Vec<_> = (0..6)
            .map(|i| {
                let mut h = [0.0; 6];
                h[i] = 1.0;
                (h, target[i], weight)
            })
            .collect();
        let f = srif_with(&rows);

        let mut x = [0.0_f64; 6];
        // SAFETY: `f` is live and `x` is six writable doubles.
        assert_eq!(unsafe { sl_srif_solve(f, x.as_mut_ptr()) }, SL_OK);
        for (got, want) in x.iter().zip(&target) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }

        let mut cov = [0.0_f64; 36];
        // SAFETY: `f` is live and `cov` is 36 writable doubles.
        assert_eq!(unsafe { sl_srif_covariance(f, cov.as_mut_ptr()) }, SL_OK);
        for i in 0..6 {
            for j in 0..6 {
                let want = if i == j { 1.0 / weight } else { 0.0 };
                assert!(
                    (cov[i * 6 + j] - want).abs() < 1e-12,
                    "covariance[{i}][{j}] = {}",
                    cov[i * 6 + j]
                );
            }
        }

        let mut sum_of_squares = -1.0;
        let mut folded = 0_u32;
        // SAFETY: `f` is live and both out-parameters are writable.
        assert_eq!(
            unsafe { sl_srif_residual_stats(f, &raw mut sum_of_squares, &raw mut folded) },
            SL_OK
        );
        assert_eq!(folded, 6);
        assert!(
            sum_of_squares.abs() < 1e-18,
            "six rows determining six unknowns leave no residual, got {sum_of_squares}"
        );

        // SAFETY: `f` came from `sl_srif_open` and has not been closed.
        unsafe { sl_srif_close(f) };
        // Closing null is documented as a no-op, and this is the assertion of it.
        // SAFETY: null is explicitly allowed.
        unsafe { sl_srif_close(core::ptr::null_mut()) };
    }

    #[test]
    fn the_information_filter_refuses_a_system_it_cannot_determine() {
        let f = srif_with(&[([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0, 1.0)]);
        let mut x = [0.0_f64; 6];
        // SAFETY: `f` is live and `x` is six writable doubles.
        assert_eq!(unsafe { sl_srif_solve(f, x.as_mut_ptr()) }, SL_ERR_DOMAIN);
        let mut cov = [0.0_f64; 36];
        // SAFETY: as above.
        assert_eq!(
            unsafe { sl_srif_covariance(f, cov.as_mut_ptr()) },
            SL_ERR_DOMAIN
        );

        // A non-finite row is refused before it can poison the triangle, so the filter that
        // follows is the one that preceded it.
        // SAFETY: `f` is live and the row is six doubles.
        assert_eq!(
            unsafe { sl_srif_add_row(f, [f64::NAN; 6].as_ptr(), 1.0, 1.0) },
            SL_ERR_INVALID_ARGUMENT
        );
        // SAFETY: as above.
        assert_eq!(
            unsafe { sl_srif_add_row(f, [1.0_f64; 6].as_ptr(), 1.0, -1.0) },
            SL_ERR_INVALID_ARGUMENT
        );
        let mut folded = 0_u32;
        let mut sum_of_squares = 0.0;
        // SAFETY: as above.
        assert_eq!(
            unsafe { sl_srif_residual_stats(f, &raw mut sum_of_squares, &raw mut folded) },
            SL_OK
        );
        assert_eq!(folded, 1, "a refused row is not a folded row");
        // SAFETY: `f` came from `sl_srif_open` and has not been closed.
        unsafe { sl_srif_close(f) };
    }

    #[test]
    fn the_information_filter_takes_a_prior_or_neither_half_of_one() {
        let x0 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // `R = 2I` is `chol(P₀⁻¹)` for `P₀ = I/4`, so a prior alone already determines the state.
        let mut r0 = [0.0_f64; 36];
        for i in 0..6 {
            r0[i * 6 + i] = 2.0;
        }
        let mut f: *mut SlSrif = core::ptr::null_mut();
        // SAFETY: both halves are the sizes documented, and `f` is one writable pointer.
        assert_eq!(
            unsafe { sl_srif_open(x0.as_ptr(), r0.as_ptr(), &raw mut f) },
            SL_OK
        );
        let mut x = [0.0_f64; 6];
        // SAFETY: `f` is live and `x` is six writable doubles.
        assert_eq!(unsafe { sl_srif_solve(f, x.as_mut_ptr()) }, SL_OK);
        assert_eq!(x, x0, "a prior of full rank is already the answer");
        let mut cov = [0.0_f64; 36];
        // SAFETY: as above.
        assert_eq!(unsafe { sl_srif_covariance(f, cov.as_mut_ptr()) }, SL_OK);
        assert!((cov[0] - 0.25).abs() < 1e-15, "P₀ = (RᵀR)⁻¹ = I/4");
        // SAFETY: `f` came from `sl_srif_open` and has not been closed.
        unsafe { sl_srif_close(f) };

        // Half a prior is a mistake rather than a shorthand.
        let mut half: *mut SlSrif = core::ptr::null_mut();
        // SAFETY: `half` is one writable pointer; the other arguments are as documented.
        assert_eq!(
            unsafe { sl_srif_open(x0.as_ptr(), core::ptr::null(), &raw mut half) },
            SL_ERR_INVALID_ARGUMENT
        );
        assert!(
            half.is_null(),
            "a refused open leaves the caller's pointer alone"
        );
        // SAFETY: as above.
        assert_eq!(
            unsafe { sl_srif_open(core::ptr::null(), r0.as_ptr(), &raw mut half) },
            SL_ERR_INVALID_ARGUMENT
        );
        // SAFETY: as above, with a non-finite prior.
        assert_eq!(
            unsafe { sl_srif_open([f64::NAN; 6].as_ptr(), r0.as_ptr(), &raw mut half) },
            SL_ERR_INVALID_ARGUMENT
        );
        // SAFETY: null is explicitly allowed for the out-pointer check.
        assert_eq!(
            unsafe { sl_srif_open(core::ptr::null(), core::ptr::null(), core::ptr::null_mut()) },
            SL_ERR_INVALID_ARGUMENT
        );
    }
    // ── Constellation families, coverage and sensors ──────────────────────────────────────────

    /// The two-call protocol, once, for the tests below: ask the length, size the buffer, fill it.
    /// `zero` is the value the buffer starts at, so that a short write is visible as one.
    fn collect<T: Copy>(
        zero: T,
        mut call: impl FnMut(*mut T, usize, *mut usize) -> i32,
    ) -> Result<Vec<T>, i32> {
        let mut len = 0_usize;
        let status = call(core::ptr::null_mut(), 0, &raw mut len);
        if status != SL_OK {
            return Err(status);
        }
        let mut out = vec![zero; len];
        let status = call(out.as_mut_ptr(), out.len(), &raw mut len);
        if status != SL_OK {
            return Err(status);
        }
        out.truncate(len);
        Ok(out)
    }

    /// The one cross-check that ties the new generator to an already-verified one: a circular
    /// lattice `(No, Nso, Nc) = (P, S, F)` *is* a Walker Delta `i: T/P/F`, satellite for satellite.
    /// The crate proves it about `LatticeFlower`; this proves the C entry point is wired to it.
    #[test]
    fn the_lattice_reduces_to_the_walker_it_contains() {
        let (planes, per_plane, phasing) = (4_u32, 6_u32, 3_u32);
        let (a, inc) = (RE_EARTH + 20_000e3, 56.0_f64.to_radians());

        let mut lattice = SlLatticeParams {
            orbits: 0,
            sats_per_orbit: 0,
            config: 0,
            semi_major_axis_m: 0.0,
            eccentricity: 0.0,
            inclination_rad: 0.0,
            arg_periapsis_rad: 0.0,
            raan0_rad: 0.0,
            mu: 0.0,
        };
        // SAFETY: the pointer is to a local of the right type.
        unsafe { sl_lattice_params_init(&raw mut lattice) };
        assert_eq!(lattice.mu, MU_EARTH, "init must supply Earth's mu");
        lattice.orbits = planes;
        lattice.sats_per_orbit = per_plane;
        lattice.config = phasing;
        lattice.semi_major_axis_m = a;
        lattice.inclination_rad = inc;

        let mut walker = SlWalkerParams {
            pattern: WALKER_DELTA,
            inclination_rad: inc,
            planes,
            per_plane,
            phasing,
            semi_major_axis_m: a,
            mu: MU_EARTH,
        };
        // SAFETY: the pointer is to a local of the right type.
        unsafe { sl_walker_params_init(&raw mut walker) };
        walker.inclination_rad = inc;
        walker.planes = planes;
        walker.per_plane = per_plane;
        walker.phasing = phasing;
        walker.semi_major_axis_m = a;

        let from_lattice = collect(state([0.0; 3], [0.0; 3]), |out, cap, len| {
            // SAFETY: `out`/`cap`/`len` are the harness's own buffer, described honestly.
            unsafe { sl_lattice_flower(&raw const lattice, out, cap, len) }
        })
        .expect("a circular lattice generates");
        let from_walker = collect(state([0.0; 3], [0.0; 3]), |out, cap, len| {
            // SAFETY: as above.
            unsafe { sl_walker(&raw const walker, out, cap, len) }
        })
        .expect("the Walker it equals generates");

        assert_eq!(from_lattice.len(), (planes * per_plane) as usize);
        for (i, (l, w)) in from_lattice.iter().zip(&from_walker).enumerate() {
            let (dr, dv) = (
                (to_vec3(l.position) - to_vec3(w.position)).norm(),
                (to_vec3(l.velocity) - to_vec3(w.velocity)).norm(),
            );
            assert!(
                dr < 1e-6 && dv < 1e-9,
                "satellite {i}: the lattice and the Walker it contains disagree by {dr} m, {dv} m/s"
            );
        }
    }

    /// The Flower phases its satellites along one shared track, and Draim's four fly the orbit its
    /// geometry describes -- checked against Kepler's third law rather than against itself.
    #[test]
    fn the_flower_and_the_draim_fly_the_orbits_they_describe() {
        let mut f = SlFlowerParams {
            petals: 0,
            days: 0,
            satellites: 0,
            eccentricity: 0.0,
            inclination_rad: 0.0,
            arg_periapsis_rad: 0.0,
            raan_rad: 0.0,
            mean_anomaly_rad: 0.0,
            frame_rate_rad_s: 0.0,
            mu: 0.0,
        };
        // SAFETY: the pointer is to a local of the right type.
        unsafe { sl_flower_params_init(&raw mut f) };
        assert_eq!(f.mu, MU_EARTH, "init must supply Earth's mu");
        assert!(
            (f.frame_rate_rad_s - 7.292_115_146_5e-5).abs() < 1e-18,
            "init must supply Earth's sidereal rate, not a zero that makes the track inertial"
        );
        f.petals = 3;
        f.days = 1;
        f.satellites = 6;
        f.eccentricity = 0.2;
        f.inclination_rad = 63.4_f64.to_radians();
        f.arg_periapsis_rad = core::f64::consts::FRAC_PI_2;

        let petals = collect(state([0.0; 3], [0.0; 3]), |out, cap, len| {
            // SAFETY: the harness's own buffer, described honestly.
            unsafe { sl_flower(&raw const f, out, cap, len) }
        })
        .expect("a three-petal Flower generates");
        assert_eq!(petals.len(), f.satellites as usize);

        // Every satellite shares one orbit: vis-viva gives the same axis for all six.
        let axis = |s: &SlState| {
            let (r, v) = (to_vec3(s.position).norm(), to_vec3(s.velocity).norm());
            1.0 / (2.0 / r - v * v / MU_EARTH)
        };
        let a0 = axis(&petals[0]);
        for (i, s) in petals.iter().enumerate() {
            let rel = (axis(s) - a0).abs() / a0;
            assert!(
                rel < 1e-12,
                "satellite {i} is not on the shared track: {rel}"
            );
        }

        let mut d = SlDraimParams {
            period_s: 0.0,
            eccentricity: 0.0,
            inclination_rad: 0.0,
            raan0_rad: 0.0,
            mu: 0.0,
        };
        // SAFETY: the pointer is to a local of the right type.
        unsafe { sl_draim_params_init(&raw mut d) };
        assert_eq!(d.period_s, 172_800.0, "the patent's 48-hour orbit");
        assert_eq!(d.eccentricity, 0.28);
        assert!(
            (d.inclination_rad - 33.0_f64.to_radians()).abs() < 1e-15,
            "the patent's 33 degrees"
        );

        let mut orbit = SlDraimOrbit {
            semi_major_axis_m: 0.0,
            perigee_radius_m: 0.0,
            apogee_radius_m: 0.0,
        };
        // SAFETY: both pointers are to locals of the right type.
        assert_eq!(
            unsafe { sl_draim_geometry(&raw const d, &raw mut orbit) },
            SL_OK
        );
        let want_a = (MU_EARTH * (d.period_s / core::f64::consts::TAU).powi(2)).cbrt();
        assert!(
            (orbit.semi_major_axis_m - want_a).abs() / want_a < 1e-12,
            "Kepler's third law says {want_a} m, the binding says {} m",
            orbit.semi_major_axis_m
        );
        assert!(
            (orbit.perigee_radius_m - want_a * (1.0 - d.eccentricity)).abs() < 1e-6
                && (orbit.apogee_radius_m - want_a * (1.0 + d.eccentricity)).abs() < 1e-6,
            "the radii must bracket the axis by the eccentricity"
        );

        let four = collect(state([0.0; 3], [0.0; 3]), |out, cap, len| {
            // SAFETY: the harness's own buffer, described honestly.
            unsafe { sl_draim(&raw const d, out, cap, len) }
        })
        .expect("Draim's design generates");
        assert_eq!(four.len(), 4, "the tetrahedral design is four satellites");
        for (i, s) in four.iter().enumerate() {
            let a = axis(s);
            assert!(
                (a - want_a).abs() / want_a < 1e-9,
                "satellite {i} is not on the design's orbit: {a} m against {want_a} m"
            );
        }
    }

    /// `sl_minimum_walker` answers with a design, not with three numbers: the result is handed
    /// straight back to `sl_walker` and the constellation it builds is measured against the
    /// footprint that was asked for.
    #[test]
    fn the_minimum_walker_is_a_design_that_covers_what_was_asked_of_it() {
        let (a, inc, footprint) = (RE_EARTH + 1_200e3, 60.0_f64.to_radians(), 0.5);
        let mut design = SlWalkerParams {
            pattern: 0,
            inclination_rad: 0.0,
            planes: 0,
            per_plane: 0,
            phasing: 0,
            semi_major_axis_m: 0.0,
            mu: 0.0,
        };
        // SAFETY: the out-pointer is to a local of the right type.
        let status = unsafe {
            sl_minimum_walker(footprint, inc, a, MU_EARTH, 1, 40, 120, 12, &raw mut design)
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert_eq!(design.pattern, WALKER_DELTA);
        assert_eq!(design.inclination_rad, inc);
        assert_eq!(design.semi_major_axis_m, a);
        assert_eq!(design.mu, MU_EARTH);
        assert!(design.planes * design.per_plane <= 40, "within the budget");

        let states = collect(state([0.0; 3], [0.0; 3]), |out, cap, len| {
            // SAFETY: the harness's own buffer, described honestly.
            unsafe { sl_walker(&raw const design, out, cap, len) }
        })
        .expect("the answer is a Walker input");
        let mut alpha = 0.0;
        // SAFETY: `states` is a live Vec of its own length; `alpha` is a local.
        let status = unsafe {
            sl_alpha_characteristic(
                states.as_ptr(),
                states.len(),
                MU_EARTH,
                1,
                120,
                12,
                &raw mut alpha,
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert!(
            alpha <= footprint,
            "the design the search returned needs {alpha} rad, more than the {footprint} asked for"
        );

        // A footprint of zero is not a search that fails, it is a search that was never posed --
        // and the two are different status codes, which is what separates a fixed call from a
        // bisected one.
        // SAFETY: the out-pointer is to a local of the right type.
        let status =
            unsafe { sl_minimum_walker(0.0, inc, a, MU_EARTH, 1, 4, 60, 6, &raw mut design) };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT, "{}", last_error());

        // A footprint no constellation of four can reach is a named domain failure, not a design.
        // SAFETY: the out-pointer is to a local of the right type.
        let status =
            unsafe { sl_minimum_walker(0.05, inc, a, MU_EARTH, 1, 4, 60, 6, &raw mut design) };
        assert_eq!(status, SL_ERR_DOMAIN);
        assert!(
            last_error().contains("at most 4 satellites"),
            "{}",
            last_error()
        );
    }

    /// The grid is the points it names: on the sphere, outward, and longitude-major within each
    /// latitude -- the order the two coverage entry points read it back in.
    #[test]
    fn the_grid_cells_are_the_points_they_name() {
        let lats = [-0.5_f64, 0.0, 0.5];
        let lons = [0.0_f64, 1.0, 2.0, 3.0];
        let cells = collect(state([0.0; 3], [0.0; 3]), |out, cap, len| {
            // SAFETY: the arrays are locals of the stated length; the buffer is the harness's own.
            unsafe {
                sl_grid_cells(
                    lats.as_ptr(),
                    lats.len(),
                    lons.as_ptr(),
                    lons.len(),
                    out,
                    cap,
                    len,
                )
            }
        })
        .expect("a 3x4 grid generates");
        assert_eq!(cells.len(), lats.len() * lons.len());

        for (k, c) in cells.iter().enumerate() {
            let (phi, lam) = (lats[k / lons.len()], lons[k % lons.len()]);
            let up = Vec3::new(phi.cos() * lam.cos(), phi.cos() * lam.sin(), phi.sin());
            let r = to_vec3(c.position);
            assert!(
                (r.norm() - RE_EARTH).abs() < 1e-6,
                "cell {k} is not on the sphere"
            );
            assert!(
                (r - up.scale(RE_EARTH)).norm() < 1e-6 && (to_vec3(c.velocity) - up).norm() < 1e-12,
                "cell {k} is not the (lat {phi}, lon {lam}) the longitude-major order says it is"
            );
        }
    }

    /// The aggregate is what it aggregates: the per-cell metrics `sl_coverage_scan` reports are
    /// the windows `sl_access_windows` lists, and the figures of merit are those metrics.
    #[test]
    fn the_coverage_metrics_are_the_windows_they_summarise() {
        let mut sensor = SlSensor {
            kind: 0,
            min_elevation_rad: 0.0,
            near_incidence_rad: 0.0,
            far_incidence_rad: 0.0,
            max_off_nadir_rad: 0.0,
        };
        // SAFETY: the pointer is to a local of the right type.
        unsafe { sl_sensor_init(&raw mut sensor, SENSOR_ELEVATION_MASK) };
        sensor.min_elevation_rad = 10.0_f64.to_radians();

        let lats = [0.0_f64, 0.4];
        let lons = [0.0_f64, 1.5];
        let cells = collect(state([0.0; 3], [0.0; 3]), |out, cap, len| {
            // SAFETY: the arrays are locals of the stated length; the buffer is the harness's own.
            unsafe {
                sl_grid_cells(
                    lats.as_ptr(),
                    lats.len(),
                    lons.as_ptr(),
                    lons.len(),
                    out,
                    cap,
                    len,
                )
            }
        })
        .expect("a 2x2 grid generates");

        let (a, inc, horizon) = (RE_EARTH + 700e3, 51.6_f64.to_radians(), 86_400.0);
        let mut metrics = vec![
            SlCoverageMetrics {
                passes: 0,
                coverage_fraction: 0.0,
                max_gap_s: 0.0,
                mean_gap_s: 0.0,
                mean_response_time_s: 0.0,
            };
            cells.len()
        ];
        // SAFETY: sensor, cells and out are live locals described by their own lengths.
        let status = unsafe {
            sl_coverage_scan(
                a,
                inc,
                0.0,
                0.0,
                &raw const sensor,
                cells.as_ptr(),
                cells.len(),
                horizon,
                1.0,
                0,
                metrics.as_mut_ptr(),
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());

        let mut any = false;
        for (k, m) in metrics.iter().enumerate() {
            let cell = cells[k];
            let windows = collect(
                SlWindow {
                    rise_s: 0.0,
                    set_s: 0.0,
                },
                |out, cap, len| {
                    // SAFETY: sensor and cell are live locals; the buffer is the harness's own.
                    unsafe {
                        sl_access_windows(
                            a,
                            inc,
                            0.0,
                            0.0,
                            &raw const sensor,
                            &raw const cell,
                            horizon,
                            1.0,
                            out,
                            cap,
                            len,
                        )
                    }
                },
            )
            .expect("the windows behind the metrics list");
            assert_eq!(
                windows.len(),
                m.passes as usize,
                "cell {k}: {} windows against {} passes",
                windows.len(),
                m.passes
            );
            let covered: f64 = windows.iter().map(|w| w.set_s - w.rise_s).sum();
            assert!(
                (m.coverage_fraction - covered / horizon).abs() < 1e-9,
                "cell {k}: the fraction is not the windows' own share of the horizon"
            );
            for w in &windows {
                assert!(
                    w.rise_s < w.set_s && w.rise_s >= 0.0 && w.set_s <= horizon,
                    "cell {k}: a window outside the scan"
                );
            }
            any |= m.passes > 0;
        }
        assert!(any, "a 700 km orbit over a day must see something");

        let mut fom = SlConstellationFom {
            point_count: 0,
            mean_coverage: 0.0,
            min_coverage: 0.0,
            worst_max_gap_s: 0.0,
            mean_max_gap_s: 0.0,
            mean_response_time_s: 0.0,
            worst_response_time_s: 0.0,
            fully_covered_points: 0,
        };
        // SAFETY: metrics is a live Vec of its own length; a null weight array is documented.
        let status = unsafe {
            sl_figures_of_merit(
                metrics.as_ptr(),
                metrics.len(),
                core::ptr::null(),
                &raw mut fom,
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert_eq!(fom.point_count, metrics.len() as u32);
        let mean = metrics.iter().map(|m| m.coverage_fraction).sum::<f64>() / metrics.len() as f64;
        let worst = metrics.iter().fold(0.0_f64, |acc, m| acc.max(m.max_gap_s));
        assert!(
            (fom.mean_coverage - mean).abs() < 1e-12 && (fom.worst_max_gap_s - worst).abs() < 1e-9,
            "the figures of merit are not the metrics they aggregate"
        );
        assert!(fom.min_coverage <= fom.mean_coverage);
    }

    /// An EOP series is measured, so the interpolation between two rows and the refusal to continue
    /// the line past the last one are both part of the answer. The propagation tests build a series
    /// that never changes -- deliberately, because that isolates the plumbing -- and a constant
    /// series agrees with every formula, so the formula itself is measured here.
    #[test]
    fn the_eop_series_is_a_straight_line_inside_and_a_held_value_outside() {
        let row = |t_seconds: f64, dut1_seconds: f64| SlEopSample {
            t_seconds,
            xp_rad: 0.0,
            yp_rad: 0.0,
            dut1_seconds,
            dx_rad: 0.0,
            dy_rad: 0.0,
        };
        let rows = [row(0.0, 1.0), row(10.0, 3.0), row(30.0, 7.0)];
        let dut1 = |t: f64| lerp_at(&rows, |s| s.dut1_seconds, t);

        // Inside: the straight line through whichever pair brackets the time, including the knots.
        for (t, want) in [
            (0.0, 1.0),
            (2.5, 1.5),
            (10.0, 3.0),
            (15.0, 4.0),
            (25.0, 6.0),
            (30.0, 7.0),
        ] {
            assert!(
                (dut1(t) - want).abs() < 1e-12,
                "at t = {t} s the series reads {}, and the line through its rows is {want}",
                dut1(t)
            );
        }

        // Outside: the end row held, not the line continued -- extrapolating a measurement invents
        // one, and the two ends are separate branches so both are asked.
        assert!((dut1(-1e6) - 1.0).abs() < 1e-12);
        assert!((dut1(1e6) - 7.0).abs() < 1e-12);
    }

    /// The seed an adaptive integrator starts from: a thousandth of the arc, never below a second,
    /// and never negative for a backwards one. It is only a seed, so nothing downstream can measure
    /// it to better than a step count -- which is why it is measured here instead.
    #[test]
    fn the_first_step_is_a_thousandth_of_the_arc_with_a_floor_under_it() {
        assert!((initial_step(86_400.0) - 86.4).abs() < 1e-12);
        assert!((initial_step(-86_400.0) - 86.4).abs() < 1e-12);
        assert!((initial_step(500.0) - 1.0).abs() < 1e-12);
        assert!((initial_step(0.0) - 1.0).abs() < 1e-12);
    }

    /// Every coverage answer is that track sampled, so nothing downstream can measure it: the pass
    /// counts and the gap statistics agree with the windows whatever curve the satellite is put on.
    /// The track therefore has to be checked against something outside the coverage engine, and the
    /// classical construction is that something -- `R3(-raan) R1(-inc) R3(-u)` on `(a, 0, 0)`, then
    /// the Earth's rotation. The source writes that triple product out expanded; this builds it as
    /// three elementary rotations, and takes the mean motion from Kepler's third law rather than
    /// from the same expression, so neither side is the other restated.
    #[test]
    fn the_ground_track_is_the_three_rotations_its_elements_name() {
        let (a, inc, raan, phase) = (7_100_000.0, 0.9, 0.7, 0.4);
        let track = circular_ground_track(a, inc, raan, phase);
        let period = core::f64::consts::TAU * (a * a * a / MU_EARTH).sqrt();

        for &t in &[0.0, 600.0, 3_600.0, 20_000.0] {
            let u = phase + core::f64::consts::TAU * t / period;
            let plane = Mat3::rot3(-u) * Vec3::new(a, 0.0, 0.0);
            let eci = Mat3::rot3(-raan) * (Mat3::rot1(-inc) * plane);
            let want = Mat3::rot3(spacelab_frames::EARTH_ROTATION_RATE * t) * eci;
            let got = track(t);
            assert!(
                (got - want).norm() < 1e-6,
                "at t = {t} s the track is at {got:?}, not the {want:?} its elements give"
            );
            assert!(
                (got.norm() - a).abs() < 1e-6,
                "a circular orbit does not change radius, and this one is at {} m",
                got.norm()
            );
        }

        // The inclination is the highest latitude the track reaches, which is a statement about the
        // orbit rather than about any one sample.
        let highest = (0..720)
            .map(|k| {
                let r = track(period * f64::from(k) / 720.0);
                (r.z / r.norm()).asin().abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(
            (highest - inc).abs() < 1e-3,
            "the track reaches {highest} rad of latitude on an orbit inclined at {inc}"
        );
    }

    /// The three sensors are three constraints, and each says what it will not accept -- a tag it
    /// does not know before it reads a field that tag would not have set.
    #[test]
    fn a_coverage_request_that_cannot_be_answered_names_what_is_wrong_with_it() {
        let cell = state([RE_EARTH, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let (a, horizon) = (RE_EARTH + 700e3, 3_600.0);
        let mut len = 0_usize;
        let mut sensor = SlSensor {
            kind: 0,
            min_elevation_rad: 0.0,
            near_incidence_rad: 0.0,
            far_incidence_rad: 0.0,
            max_off_nadir_rad: 0.0,
        };

        let mut windows = |s: &SlSensor, a: f64, horizon: f64, tol: f64| {
            // SAFETY: every pointer is to a live local; sizing with a null buffer is documented.
            unsafe {
                sl_access_windows(
                    a,
                    0.9,
                    0.0,
                    0.0,
                    &raw const *s,
                    &raw const cell,
                    horizon,
                    tol,
                    core::ptr::null_mut(),
                    0,
                    &raw mut len,
                )
            }
        };

        // SAFETY: the pointer is to a local of the right type.
        unsafe { sl_sensor_init(&raw mut sensor, 7) };
        assert_eq!(windows(&sensor, a, horizon, 1.0), SL_ERR_INVALID_ARGUMENT);
        assert!(last_error().contains("sensor tag 7"), "{}", last_error());

        // SAFETY: the pointer is to a local of the right type.
        unsafe { sl_sensor_init(&raw mut sensor, SENSOR_SIDE_LOOKING_SAR) };
        sensor.near_incidence_rad = 0.8;
        sensor.far_incidence_rad = 0.4;
        assert_eq!(
            windows(&sensor, a, horizon, 1.0),
            SL_ERR_DOMAIN,
            "a SAR whose near edge is past its far one is not a constraint"
        );

        // SAFETY: the pointer is to a local of the right type.
        unsafe { sl_sensor_init(&raw mut sensor, SENSOR_AGILE_OPTICAL) };
        sensor.max_off_nadir_rad = 0.4;
        assert_eq!(windows(&sensor, -a, horizon, 1.0), SL_ERR_INVALID_ARGUMENT);
        assert!(last_error().contains("semi-major axis"), "{}", last_error());
        // Zero is not a small orbit, it is no orbit, and the bound has to be strict to say so.
        assert_eq!(windows(&sensor, 0.0, horizon, 1.0), SL_ERR_INVALID_ARGUMENT);
        assert!(last_error().contains("semi-major axis"), "{}", last_error());
        assert_eq!(windows(&sensor, a, 0.0, 1.0), SL_ERR_INVALID_ARGUMENT);
        assert!(last_error().contains("horizon"), "{}", last_error());
        assert_eq!(windows(&sensor, a, horizon, 0.0), SL_ERR_INVALID_ARGUMENT);
        assert!(last_error().contains("tol"), "{}", last_error());

        // The three orbit angles are checked in one condition, and any one of them alone has to be
        // enough to refuse -- a NaN anywhere is a NaN everywhere on the track.
        for (inc, raan, phase) in [
            (f64::NAN, 0.0, 0.0),
            (0.9, f64::NAN, 0.0),
            (0.9, 0.0, f64::NAN),
        ] {
            let mut refused = 0_usize;
            // SAFETY: every pointer is to a live local; sizing with a null buffer is documented.
            let status = unsafe {
                sl_access_windows(
                    a,
                    inc,
                    raan,
                    phase,
                    &raw const sensor,
                    &raw const cell,
                    horizon,
                    1.0,
                    core::ptr::null_mut(),
                    0,
                    &raw mut refused,
                )
            };
            assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
            assert!(last_error().contains("orbit angles"), "{}", last_error());
        }

        assert_eq!(windows(&sensor, a, horizon, 1.0), SL_OK, "{}", last_error());

        // The α-characteristic and the aggregate refuse an empty population rather than inventing
        // an answer for it.
        let mut alpha = 0.0;
        // SAFETY: `cell` is a live local; a count of zero is what is being rejected.
        assert_eq!(
            unsafe {
                sl_alpha_characteristic(&raw const cell, 0, MU_EARTH, 1, 10, 10, &raw mut alpha)
            },
            SL_ERR_INVALID_ARGUMENT
        );
        // SAFETY: as above, with a mu that cannot be a gravitational parameter.
        assert_eq!(
            unsafe { sl_alpha_characteristic(&raw const cell, 1, 0.0, 1, 10, 10, &raw mut alpha) },
            SL_ERR_INVALID_ARGUMENT
        );
        // SAFETY: as above, with a fold nobody can cover zero times.
        assert_eq!(
            unsafe {
                sl_alpha_characteristic(&raw const cell, 1, MU_EARTH, 0, 10, 10, &raw mut alpha)
            },
            SL_ERR_INVALID_ARGUMENT
        );

        // A non-finite position is refused whichever component carries it: `from_state` answers a
        // NaN element set rather than refusing, and a NaN element set covers nothing while
        // reporting a number. Each axis is asked separately, because one condition that reads all
        // three passes a test that only ever makes one of them fail.
        for axis in 0..3 {
            let mut broken = cell;
            match axis {
                0 => broken.position.x = f64::NAN,
                1 => broken.position.y = f64::NAN,
                _ => broken.position.z = f64::INFINITY,
            }
            // SAFETY: `broken` is a live local; the non-finite component is what is being rejected.
            assert_eq!(
                unsafe {
                    sl_alpha_characteristic(
                        &raw const broken,
                        1,
                        MU_EARTH,
                        1,
                        10,
                        10,
                        &raw mut alpha,
                    )
                },
                SL_ERR_INVALID_ARGUMENT,
                "a non-finite component {axis} of the position was accepted"
            );
        }
    }
    // ── GNSS, ionosphere and troposphere ──────────────────────────────────────────────────────

    /// A RINEX-3 `D`-exponent field, spec columns.
    fn d19(x: f64) -> String {
        let text = format!("{x:.12E}");
        let (mantissa, exponent) = text.split_once('E').expect("scientific notation has an E");
        let exponent: i32 = exponent.parse().expect("the exponent is an integer");
        format!(
            "{sign}{mantissa}D{esign}{exponent:02}",
            sign = if x < 0.0 { "-" } else { " " },
            mantissa = mantissa.trim_start_matches('-'),
            esign = if exponent >= 0 { "+" } else { "-" },
            exponent = exponent.abs(),
        )
    }

    /// One broadcast-orbit line. The indent is **four** spaces -- `4X` in the format's own notation
    /// -- and that is worth a comment because it used to be three here, in the C ABI, in the Python
    /// bridge and in the reader itself: four fixtures agreeing with the parser about the columns
    /// instead of with the format, so none of them could disagree with it about a value.
    fn orbit_line(a: f64, b: f64, c: f64, d: f64) -> String {
        format!("    {}{}{}{}", d19(a), d19(b), d19(c), d19(d))
    }

    /// A one-record RINEX-3 navigation file: a GPS record and a GLONASS one, the same fixture the
    /// Python bridge's smoke test uses, so the two bindings are read against the same bytes.
    fn rinex_fixture() -> String {
        let header = format!(
            "     3.04           RINEX VERSION / TYPE\n{}END OF HEADER\n",
            " ".repeat(60)
        );
        let gps = format!(
            "G07 2024 01 09 12 00 00{}{}{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n",
            d19(1.5e-4),
            d19(3.4e-12),
            d19(0.0),
            orbit_line(1.0, 35.0, 4.5e-9, 0.321),
            orbit_line(1.2e-6, 0.0062, 8.3e-6, 5153.65),
            orbit_line(216_000.0, -1.1e-7, -1.234, 9.0e-8),
            orbit_line(0.9599, 210.0, 0.567, -8.1e-9),
            orbit_line(-2.1e-10, 1.0, 2300.0, 0.0),
            orbit_line(0.0, 0.0, -1.0e-8, 40.0),
            orbit_line(216_000.0, 4.0, 0.0, 0.0),
        );
        let glonass = format!(
            "R05 2024 01 09 03 15 00{}{}{}\n{}\n{}\n{}\n",
            d19(-1.5e-4),
            d19(0.0),
            d19(11_700.0),
            orbit_line(25_510.0, 0.1, 1.0e-9, 0.0),
            orbit_line(-1200.0, 2.9, -2.0e-9, -1.0),
            orbit_line(800.0, 1.3, 3.0e-9, 0.0),
        );
        format!("{header}{gps}{glonass}")
    }

    /// The two RINEX entry points read the same file and each take only their own constellation
    /// out of it -- which is the reason the GNSS one carries a system tag at all.
    #[test]
    fn the_rinex_bindings_take_only_their_own_constellation_from_a_mixed_file() {
        let text = rinex_fixture();
        let zero = SlGnssState {
            system: 0,
            prn: 0,
            state: state([0.0; 3], [0.0; 3]),
        };
        let sats = collect(zero, |out, cap, len| {
            // SAFETY: `text` is a live String; the buffer is the harness's own.
            unsafe { sl_rinex_gnss_positions(text.as_ptr().cast(), text.len(), 0.0, out, cap, len) }
        })
        .expect("the fixture holds one GPS record");
        assert_eq!(sats.len(), 1, "the GLONASS record is not a GNSS one");
        assert_eq!(sats[0].system, GNSS_GPS);
        assert_eq!(sats[0].prn, 7);
        let r = to_vec3(sats[0].state.position).norm();
        assert!(
            (26.0e6..27.2e6).contains(&r),
            "sqrt(a) = 5153.65 is a medium Earth orbit, not {r} m"
        );

        let zero = SlGlonassState {
            slot: 0,
            state: state([0.0; 3], [0.0; 3]),
        };
        let glonass = collect(zero, |out, cap, len| {
            // SAFETY: as above.
            unsafe {
                sl_rinex_glonass_positions(text.as_ptr().cast(), text.len(), 0.0, out, cap, len)
            }
        })
        .expect("the fixture holds one GLONASS record");
        assert_eq!(glonass.len(), 1, "the GPS record is not a GLONASS one");
        assert_eq!(glonass[0].slot, 5);
        let r = to_vec3(glonass[0].state.position).norm();
        assert!(
            (24.5e6..26.5e6).contains(&r),
            "the broadcast PZ-90 state is at 25 510 km, not {r} m"
        );

        // A file with none of that kind is an empty answer, not a failure.
        let header = "     3.04           RINEX VERSION / TYPE\n";
        let mut len = 0_usize;
        // SAFETY: the string is a live literal; sizing with a null buffer is documented.
        let status = unsafe {
            sl_rinex_glonass_positions(
                header.as_ptr().cast(),
                header.len(),
                0.0,
                core::ptr::null_mut(),
                0,
                &raw mut len,
            )
        };
        assert_eq!(status, SL_OK);
        assert_eq!(len, 0, "a header with no records is zero records");

        // Bytes that are not UTF-8 are refused rather than lossily decoded.
        let invalid = [0xff_u8, 0xfe];
        // SAFETY: the array is a live local of the stated length.
        let status = unsafe {
            sl_rinex_gnss_positions(
                invalid.as_ptr().cast(),
                invalid.len(),
                0.0,
                core::ptr::null_mut(),
                0,
                &raw mut len,
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
        assert!(last_error().contains("not UTF-8"), "{}", last_error());
    }

    /// A minimal well-formed RINEX 3 observation file, written column by column. Generating it from
    /// the reader would make it agree with the reader's own mistakes.
    fn rinex_obs_fixture() -> String {
        let pad = |body: &str, label: &str| format!("{body:<60}{label}");
        [
            pad(
                "     3.05           OBSERVATION DATA    M: MIXED",
                "RINEX VERSION / TYPE",
            ),
            pad("G    2 C1C L1C", "SYS / # / OBS TYPES"),
            pad("R    1 C1C", "SYS / # / OBS TYPES"),
            pad(
                "  4460894.9000  2682361.5000 -3674748.5000",
                "APPROX POSITION XYZ",
            ),
            pad("    30.000", "INTERVAL"),
            pad("END", "END OF HEADER"),
            "> 2024 01 09 12 00  0.0000000  0  3".into(),
            "G07  23629347.915 6 124144623.325 7".into(),
            "R05  21000000.000".into(),
            "G09  23626000.000 5 124130000.000 6".into(),
            "> 2024 01 09 12 00 30.0000000  0  1".into(),
            "G07  23629400.00016 124144900.000 7".into(),
        ]
        .join("\n")
    }

    /// A fixed C field as a `String`, stopping at the terminator the writer promises.
    fn read_fixed(field: &[c_char]) -> String {
        field
            .iter()
            .take_while(|&&b| b != 0)
            .map(|&b| char::from(b.cast_unsigned()))
            .collect()
    }

    /// #312's observation reader through the ABI. Nothing about a value names it except its
    /// position in the constellation's declared descriptor list, so what is pinned here is that the
    /// position is honoured on *both* halves of the satellite identifier — the fixture puts two GPS
    /// satellites in one epoch, which is the case a match on either half alone cannot tell apart.
    #[test]
    fn the_observation_bindings_read_a_value_out_of_its_declared_slot() {
        let text = rinex_obs_fixture();
        let mut header = SlObsHeader {
            version: 0.0,
            interval: 0.0,
            approx_position: [0.0; 3],
            antenna_delta_hen: [0.0; 3],
            time_system: -1,
            file_system: 0,
            marker_name: [0; MARKER_NAME_BYTES],
        };
        // SAFETY: `text` is a live String and `header` a live local.
        let status =
            unsafe { sl_rinex_obs_header(text.as_ptr().cast(), text.len(), &raw mut header) };
        assert_eq!(status, SL_OK);
        assert!((header.version - 3.05).abs() < 1e-12, "{}", header.version);
        assert_eq!(char::from(header.file_system.cast_unsigned()), 'M');
        assert_eq!(read_fixed(&header.marker_name), "");
        assert!(
            (header.interval - 30.0).abs() < 1e-12,
            "{}",
            header.interval
        );
        assert_eq!(
            header.approx_position,
            [4_460_894.9, 2_682_361.5, -3_674_748.5]
        );
        // The header declares no antenna eccentricity, and absent is NaN in all three components.
        assert!(header.antenna_delta_hen.iter().all(|c| c.is_nan()));
        // No `TIME OF FIRST OBS` either, and the format's default is GPS rather than "unknown".
        assert_eq!(header.time_system, TIME_SYSTEM_GPS);

        let zero = SlObsDescriptor {
            code: [0; OBS_CODE_BYTES],
        };
        let codes = collect(zero, |out, cap, len| {
            // SAFETY: `text` is a live String; the buffer is the harness's own.
            unsafe {
                sl_rinex_obs_descriptors(
                    text.as_ptr().cast(),
                    text.len(),
                    b'G'.cast_signed(),
                    out,
                    cap,
                    len,
                )
            }
        })
        .expect("the fixture declares two GPS descriptors");
        assert_eq!(
            codes
                .iter()
                .map(|c| read_fixed(&c.code))
                .collect::<Vec<_>>(),
            ["C1C", "L1C"]
        );
        let absent = collect(zero, |out, cap, len| {
            // SAFETY: as above.
            unsafe {
                sl_rinex_obs_descriptors(
                    text.as_ptr().cast(),
                    text.len(),
                    b'E'.cast_signed(),
                    out,
                    cap,
                    len,
                )
            }
        })
        .expect("a constellation the file never mentions is empty, not a failure");
        assert!(absent.is_empty());

        let series = |system: u8, prn: u32, code: &str| {
            let code = std::ffi::CString::new(code).expect("a descriptor holds no interior NUL");
            let zero = SlObsSample {
                seconds_j2000: 0.0,
                value: 0.0,
                lost_lock: 0,
            };
            collect(zero, |out, cap, len| {
                // SAFETY: `text` and `code` are live for the call; the buffer is the harness's own.
                unsafe {
                    sl_rinex_obs_series(
                        text.as_ptr().cast(),
                        text.len(),
                        system.cast_signed(),
                        prn,
                        code.as_ptr(),
                        out,
                        cap,
                        len,
                    )
                }
            })
        };

        // G07 is observed at both epochs, and only the second record closes the quality digits up
        // -- `00016` is the value, then LLI 1 -- so the receiver declares a slip there and nowhere
        // else.
        let gps = series(b'G', 7, "C1C").expect("G declares C1C");
        assert_eq!(gps.len(), 2);
        assert!(
            (gps[0].value - 23_629_347.915).abs() < 1e-6,
            "{}",
            gps[0].value
        );
        assert!(
            (gps[1].value - 23_629_400.0).abs() < 1e-6,
            "{}",
            gps[1].value
        );
        assert!((gps[1].seconds_j2000 - gps[0].seconds_j2000 - 30.0).abs() < 1e-9);
        assert_eq!((gps[0].lost_lock, gps[1].lost_lock), (0, 1));

        // The second slot is a different quantity in a different unit, read out of the same records.
        let phase = series(b'G', 7, "L1C").expect("G declares L1C");
        assert!(
            (phase[0].value - 124_144_623.325).abs() < 1e-6,
            "{}",
            phase[0].value
        );

        // G09 sits behind G07 in the same epoch, and R05 shares its PRN with nothing: between them
        // they refuse a match that stopped at either half of the identifier.
        let ninth = series(b'G', 9, "C1C").expect("G declares C1C");
        assert_eq!(ninth.len(), 1);
        assert!(
            (ninth[0].value - 23_626_000.0).abs() < 1e-6,
            "{}",
            ninth[0].value
        );
        let glonass = series(b'R', 5, "C1C").expect("R declares C1C");
        assert_eq!(glonass.len(), 1);
        assert!(
            (glonass[0].value - 21_000_000.0).abs() < 1e-6,
            "{}",
            glonass[0].value
        );

        // GLONASS declares no L1C here. That is "asked for the wrong thing", not "not observed" --
        // and a satellite the file never carries is the second of those, so it is an empty series.
        assert_eq!(series(b'R', 5, "L1C").unwrap_err(), SL_ERR_INVALID_ARGUMENT);
        assert!(last_error().contains("declares no"), "{}", last_error());
        assert!(series(b'G', 31, "C1C").expect("G declares C1C").is_empty());

        // A PRN past the byte the format writes it in, a constellation that is not a letter, and
        // text that is not an observation file are each refused rather than answered.
        assert_eq!(
            series(b'G', 300, "C1C").unwrap_err(),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(series(b'7', 7, "C1C").unwrap_err(), SL_ERR_INVALID_ARGUMENT);
        assert!(last_error().contains("is a letter"), "{}", last_error());
        // SAFETY: the literal is live for the call and `header` is a live local.
        let status =
            unsafe { sl_rinex_obs_header("not a RINEX file".as_ptr().cast(), 16, &raw mut header) };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
        // SAFETY: a null out pointer is the case under test.
        let status =
            unsafe { sl_rinex_obs_header(text.as_ptr().cast(), text.len(), core::ptr::null_mut()) };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
    }

    /// The header fields the fixture above leaves absent, and the scales the enum names. Both are
    /// unreachable from a file that declares neither, and both are what a C caller switches on.
    #[test]
    fn the_observation_header_reports_the_records_it_declares() {
        let pad = |body: &str, label: &str| format!("{body:<60}{label}");
        let text = [
            pad(
                "     3.05           OBSERVATION DATA    R: GLONASS",
                "RINEX VERSION / TYPE",
            ),
            pad(
                "ONSA                                            ",
                "MARKER NAME",
            ),
            pad("R    1 C1C", "SYS / # / OBS TYPES"),
            pad(
                "        0.0010        0.0020        0.0030",
                "ANTENNA: DELTA H/E/N",
            ),
            pad(
                "  2024     1     9    12     0    0.0000000     GLO",
                "TIME OF FIRST OBS",
            ),
            pad("END", "END OF HEADER"),
        ]
        .join("\n");
        let mut header = SlObsHeader {
            version: 0.0,
            interval: 0.0,
            approx_position: [0.0; 3],
            antenna_delta_hen: [0.0; 3],
            time_system: -1,
            file_system: 0,
            marker_name: [0; MARKER_NAME_BYTES],
        };
        // SAFETY: `text` is a live String and `header` a live local.
        let status =
            unsafe { sl_rinex_obs_header(text.as_ptr().cast(), text.len(), &raw mut header) };
        assert_eq!(status, SL_OK);
        assert_eq!(read_fixed(&header.marker_name), "ONSA");
        assert_eq!(char::from(header.file_system.cast_unsigned()), 'R');
        assert_eq!(header.antenna_delta_hen, [0.001, 0.002, 0.003]);
        assert_eq!(header.time_system, TIME_SYSTEM_GLO);
        // This file declares no INTERVAL and no APPROX POSITION, and absent is NaN either way.
        assert!(header.interval.is_nan());
        assert!(header.approx_position.iter().all(|c| c.is_nan()));

        // The eight scales are distinct codes, and the enum is what a `switch` in C leans on.
        let codes = [
            TimeSystem::Gps,
            TimeSystem::Glo,
            TimeSystem::Gal,
            TimeSystem::Tai,
            TimeSystem::Utc,
            TimeSystem::Qzs,
            TimeSystem::Bdt,
            TimeSystem::Irn,
        ]
        .map(time_system_code);
        assert_eq!(codes, [0, 1, 2, 3, 4, 5, 6, 7]);

        // A column wider than the byte it is written into is refused rather than truncated onto a
        // constellation letter that means something else.
        assert_eq!(ascii_column('G'), b'G'.cast_signed());
        assert_eq!(ascii_column('é'), 0);
    }

    /// The FDMA frequencies are a closed form, so they are checked against it rather than against
    /// a table; and a channel the standard's signed byte cannot hold is NaN, not a wrap.
    #[test]
    fn the_glonass_frequencies_are_the_channel_they_name() {
        for k in -7_i32..=6 {
            let (l1, l2) = (sl_glonass_l1_frequency(k), sl_glonass_l2_frequency(k));
            assert!(
                (l1 - (1_602.0 + f64::from(k) * 0.5625) * 1e6).abs() < 1e-6,
                "L1 at {k}"
            );
            assert!(
                (l2 - (1_246.0 + f64::from(k) * 0.4375) * 1e6).abs() < 1e-6,
                "L2 at {k}"
            );
        }
        assert!(sl_glonass_l1_frequency(1_000).is_nan());
        assert!(sl_glonass_l2_frequency(-1_000).is_nan());
    }

    /// The five DOP figures are one covariance seen five ways, and the two identities between them
    /// hold exactly. Checking those rather than a pinned number means the test says something
    /// about the answer instead of about the geometry it was measured on.
    #[test]
    fn the_dop_figures_are_the_one_covariance_they_come_from() {
        let receiver = SlVec3 {
            x: RE_EARTH,
            y: 0.0,
            z: 0.0,
        };
        let a = RE_EARTH + 20_200e3;
        let satellites: Vec<SlVec3> = [(0.0, 0.0), (0.9, 0.0), (0.9, 2.1), (0.9, 4.2), (1.2, 1.0)]
            .iter()
            .map(|&(theta, phi): &(f64, f64)| SlVec3 {
                x: a * theta.cos(),
                y: a * theta.sin() * phi.cos(),
                z: a * theta.sin() * phi.sin(),
            })
            .collect();

        let mut d = SlDop {
            gdop: 0.0,
            pdop: 0.0,
            hdop: 0.0,
            vdop: 0.0,
            tdop: 0.0,
        };
        // SAFETY: every pointer is to a live local, described by its own length.
        let status = unsafe {
            sl_dilution_of_precision(
                &raw const receiver,
                satellites.as_ptr(),
                satellites.len(),
                &raw mut d,
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert!(
            (d.gdop * d.gdop - (d.pdop * d.pdop + d.tdop * d.tdop)).abs() < 1e-9,
            "GDOP^2 must be PDOP^2 + TDOP^2"
        );
        assert!(
            (d.pdop * d.pdop - (d.hdop * d.hdop + d.vdop * d.vdop)).abs() < 1e-9,
            "PDOP^2 must be HDOP^2 + VDOP^2"
        );
        assert!(d.gdop > 0.0 && d.gdop < 100.0, "gdop {}", d.gdop);

        // Three satellites cannot determine four unknowns, and the message says which.
        // SAFETY: as above, with a length of three.
        let status = unsafe {
            sl_dilution_of_precision(&raw const receiver, satellites.as_ptr(), 3, &raw mut d)
        };
        assert_eq!(status, SL_ERR_DOMAIN);
        assert!(last_error().contains("four is the"), "{}", last_error());
    }

    /// The delay models, each against the closed form or the documented floor it is defined by.
    #[test]
    fn the_delay_models_are_the_formulas_they_implement() {
        // First order is a formula, not a model: 40.3e16 * TEC / f^2, exactly.
        let l1 = 1_575.42e6;
        let delay = sl_ionospheric_path_delay(20.0, l1);
        assert!(
            (delay - 40.3e16 * 20.0 / (l1 * l1)).abs() < 1e-12,
            "the first-order delay is its own closed form"
        );
        assert!(
            (2.0..5.0).contains(&delay),
            "20 TECU on L1 is a few metres, not {delay}"
        );
        // And it falls as 1/f^2, which is the whole reason a dual-frequency receiver works.
        let l2 = 1_227.60e6;
        assert!((sl_ionospheric_path_delay(20.0, l2) / delay - (l1 * l1) / (l2 * l2)).abs() < 1e-9);

        // Klobuchar's night floor is 5 ns of group delay times the obliquity factor.
        let (alpha, beta) = ([0.0_f64; 4], [0.0_f64; 4]);
        let mut night = 0.0;
        // SAFETY: both coefficient arrays are live locals of four doubles; `night` is a local.
        let status = unsafe {
            sl_klobuchar_iono_delay(
                alpha.as_ptr(),
                beta.as_ptr(),
                0.0,
                0.0,
                core::f64::consts::FRAC_PI_2,
                0.0,
                0.0,
                &raw mut night,
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        // With every coefficient zero the amplitude term vanishes and only the floor is left: 5 ns
        // of group delay through the obliquity factor `F = 1 + 16*(0.53 - E)^3`, E in semicircles.
        // Asserting the product rather than the floor alone is what pins `F` as well.
        const C: f64 = 299_792_458.0;
        let obliquity = 16.0_f64.mul_add((0.53 - 0.5_f64).powi(3), 1.0);
        assert!(
            (night - 5e-9 * C * obliquity).abs() < 1e-6,
            "with zero coefficients only the 5 ns floor times F is left, not {night} m"
        );

        // The obliquity factor is 1 at the zenith and grows towards the horizon, monotonically.
        let zenith = sl_ionospheric_mapping_factor(core::f64::consts::FRAC_PI_2, 350e3, RE_EARTH);
        assert!(
            (zenith - 1.0).abs() < 1e-12,
            "at the zenith the ray is vertical"
        );
        let low = sl_ionospheric_mapping_factor(0.2, 350e3, RE_EARTH);
        assert!(
            low > 2.0 && low > zenith,
            "a low ray crosses more shell: {low}"
        );

        // The second-order term is small, signed and falls as 1/f^3 -- the three properties that
        // make it worth having and easy to get wrong.
        let hoi2 = |stec: f64, f: f64| {
            sl_second_order_ionospheric_delay(0.6, 0.1, 0.5, 1.0, stec, f, 350e3)
        };
        let (at_l1, at_l2) = (hoi2(50.0, l1), hoi2(50.0, l2));
        assert!(
            (1e-3..1e-2).contains(&at_l1.abs()),
            "at 50 TECU on L1 the second-order term is millimetres, not {at_l1} m"
        );
        assert!(
            (at_l2 / at_l1 - (l1 * l1 * l1) / (l2 * l2 * l2)).abs() < 1e-6,
            "the second-order term falls as 1/f^3"
        );
        // It is linear in the electron content, which is what says the TECU the caller passes is
        // converted rather than offset: a band alone would accept a scaled-down count.
        assert!(
            (hoi2(100.0, l1) / at_l1 - 2.0).abs() < 1e-12,
            "twice the electron content is twice the delay"
        );
    }

    /// The mapping functions all answer roughly 1 at the zenith, all grow towards the horizon, and
    /// all refuse an elevation at or below it -- where the ratio they compute does not exist.
    #[test]
    fn every_mapping_function_is_one_at_the_zenith_and_refuses_the_horizon() {
        let zero = SlMapping {
            hydrostatic: 0.0,
            wet: 0.0,
        };
        let (lat, lon, height, doy) = (0.7, 0.2, 200.0, 90.0);
        let zenith = core::f64::consts::FRAC_PI_2;

        let at = |elevation: f64| -> Vec<(&'static str, i32, SlMapping)> {
            let mut out = Vec::new();
            let mut m = zero;
            // SAFETY: every out-pointer is to a live local of the right type.
            unsafe {
                out.push((
                    "niell",
                    sl_niell_mapping(lat, height, doy, elevation, &raw mut m),
                    m,
                ));
                out.push((
                    "gmf",
                    sl_gmf_mapping(lat, lon, height, doy, elevation, &raw mut m),
                    m,
                ));
                out.push((
                    "vmf1",
                    sl_vmf1_mapping(1.2e-3, 6e-4, lat, height, doy, elevation, &raw mut m),
                    m,
                ));
                out.push((
                    "vmf3",
                    sl_vmf3_mapping(1.2e-3, 6e-4, lat, lon, height, doy, elevation, &raw mut m),
                    m,
                ));
            }
            out
        };

        for (name, status, m) in at(zenith) {
            assert_eq!(status, SL_OK, "{name}: {}", last_error());
            assert!(
                (m.hydrostatic - 1.0).abs() < 1e-2 && (m.wet - 1.0).abs() < 1e-2,
                "{name} at the zenith is {m:?}, and a mapping function there is 1"
            );
        }
        for ((name, status, low), (_, _, high)) in at(0.15).into_iter().zip(at(zenith)) {
            assert_eq!(status, SL_OK, "{name}: {}", last_error());
            assert!(
                low.hydrostatic > 3.0 * high.hydrostatic,
                "{name} at 8.6 degrees maps far more atmosphere than at the zenith"
            );
        }
        for (name, status, _) in at(0.0) {
            assert_eq!(
                status, SL_ERR_DOMAIN,
                "{name} must refuse the horizon rather than answer for it"
            );
        }

        // Saastamoinen's zenith delays are the two halves everything above is multiplied by.
        let dry = sl_saastamoinen_zenith_hydrostatic(1013.25, 0.7, 0.0);
        assert!(
            (2.2..2.4).contains(&dry),
            "the dry zenith delay at sea level is about 2.3 m, not {dry}"
        );
        let wet = sl_saastamoinen_zenith_wet(288.15, 15.0);
        assert!(
            (0.05..0.5).contains(&wet),
            "the wet zenith delay is decimetres, not {wet} m"
        );
    }

    /// The two NeQuick variants and the magnetic field they are checked beside: each against what
    /// the physics bounds it to, because a pinned digit here would only restate the port.
    #[test]
    fn the_electron_content_and_the_field_are_within_what_physics_allows() {
        let receiver = SlGeodetic {
            latitude_rad: 0.7,
            longitude_rad: 0.2,
            height_m: 100.0,
        };
        let satellite = SlGeodetic {
            latitude_rad: 0.9,
            longitude_rad: 0.6,
            height_m: 20_200e3,
        };
        let alpha = [80.0_f64, 0.0, 0.0];
        let mut galileo = 0.0;
        // SAFETY: every pointer is to a live local; `alpha` is three doubles.
        let status = unsafe {
            sl_nequick_galileo_stec(
                alpha.as_ptr(),
                4,
                43_200.0,
                &raw const receiver,
                &raw const satellite,
                &raw mut galileo,
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert!(
            (1.0..500.0).contains(&galileo),
            "a daytime slant TEC is tens to hundreds of TECU, not {galileo}"
        );

        // A month outside the calendar is refused before the profile is integrated.
        // SAFETY: as above, with a month that does not exist.
        let status = unsafe {
            sl_nequick_galileo_stec(
                alpha.as_ptr(),
                13,
                43_200.0,
                &raw const receiver,
                &raw const satellite,
                &raw mut galileo,
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
        assert!(last_error().contains("month 13"), "{}", last_error());

        // The clock is a second of a UTC day, so a value outside one is refused too -- and each
        // coordinate of each of the two points is checked on its own, because a condition that
        // reads three fields can lose two of them and still look right on the third.
        let nan = f64::NAN;
        let bad_points = [
            (86_401.0, receiver, satellite, "seconds_in_utc_day"),
            (
                43_200.0,
                SlGeodetic {
                    latitude_rad: nan,
                    ..receiver
                },
                satellite,
                "receiver",
            ),
            (
                43_200.0,
                SlGeodetic {
                    longitude_rad: nan,
                    ..receiver
                },
                satellite,
                "receiver",
            ),
            (
                43_200.0,
                receiver,
                SlGeodetic {
                    height_m: nan,
                    ..satellite
                },
                "satellite",
            ),
        ];
        for (seconds, rx, sat, named) in bad_points {
            // SAFETY: as above, with the one argument under test made impossible.
            let status = unsafe {
                sl_nequick_galileo_stec(
                    alpha.as_ptr(),
                    4,
                    seconds,
                    &raw const rx,
                    &raw const sat,
                    &raw mut galileo,
                )
            };
            assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
            assert!(last_error().contains(named), "{}", last_error());
        }

        // IGRF at the surface: the field is tens of microtesla everywhere on Earth.
        let position = SlVec3 {
            x: RE_EARTH * 0.6_f64.cos(),
            y: 0.0,
            z: RE_EARTH * 0.6_f64.sin(),
        };
        let mut b = SlVec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        // SAFETY: both pointers are to live locals of the right type.
        let status = unsafe { sl_igrf_field_ecef(&raw const position, 2026.0, &raw mut b) };
        assert_eq!(status, SL_OK, "{}", last_error());
        let magnitude = to_vec3(b).norm();
        assert!(
            (20e-6..70e-6).contains(&magnitude),
            "the surface field is 20-70 uT, not {magnitude} T"
        );
    }

    /// The CelesTrak space-weather product, as vendored for the propagation crate's own tests. The
    /// C ABI reads the same bytes the Python bridge does, so a divergence between the two is a
    /// difference in the binding rather than in the file.
    const SW_CSV: &str =
        include_str!("../../spacelab-propagation/tests/fixtures/celestrak_sw_sample.csv");

    /// A real CHAMP accelerometer-derived density file, 30 observations.
    const CHAMP: &str =
        include_str!("../../spacelab-propagation/tests/fixtures/champ_density_sample.txt");

    /// A 51.6-degree circular orbit at 7000 km, built rather than tabulated so the invariants below
    /// are checked against the geometry that produced it.
    fn circular_state(radius_m: f64, inclination_rad: f64) -> SlState {
        let speed = (MU_EARTH / radius_m).sqrt();
        SlState {
            position: SlVec3 {
                x: radius_m,
                y: 0.0,
                z: 0.0,
            },
            velocity: SlVec3 {
                x: 0.0,
                y: speed * inclination_rad.cos(),
                z: speed * inclination_rad.sin(),
            },
        }
    }

    /// The right ascension of the ascending node a Cartesian state carries.
    fn raan_of(s: SlState) -> f64 {
        let r = [s.position.x, s.position.y, s.position.z];
        let v = [s.velocity.x, s.velocity.y, s.velocity.z];
        let h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ];
        // n = z x h, whose components are (-h_y, h_x, 0).
        h[0].atan2(-h[1])
    }

    #[test]
    fn the_thermosphere_thins_with_height_and_answers_in_si() {
        let at = |altitude_km: f64| {
            sl_nrlmsise00_density(172, 29_000.0, altitude_km, 0.0, 0.0, 150.0, 150.0, 4.0)
        };
        let (low, mid, high) = (at(200.0), at(400.0), at(600.0));
        assert!(low > mid && mid > high, "{low} {mid} {high}");
        // kg/m³, not the model's own g/cm³: 400 km at moderate flux is a few times 1e-12.
        assert!((1e-13..1e-11).contains(&mid), "{mid}");
        // The model itself answers a sea-level 2.9e-3 kg/m³ for a NaN altitude; the binding
        // refuses rather than passing that on.
        assert!(at(f64::NAN).is_nan());
    }

    /// The binding computes the local solar time the model needs -- `sec/3600 + longitude/15` --
    /// because the model takes it as a separate input and the caller has no reason to. Nothing in
    /// the height profile above depends on it, so it is measured by the one thing it decides: where
    /// in the *universal* day the diurnal bulge sits, which moves west with the longitude at four
    /// minutes a degree.
    #[test]
    fn the_diurnal_bulge_sits_where_the_local_solar_time_puts_it() {
        // The peak of the day, in hours UT, and how far the density swings across that day.
        let day = |longitude_deg: f64| {
            let mut peak = (0.0_f64, f64::MIN);
            let mut trough = f64::MAX;
            for k in 0..48 {
                let sec = f64::from(k) * 1_800.0;
                let d =
                    sl_nrlmsise00_density(172, sec, 400.0, 0.0, longitude_deg, 150.0, 150.0, 4.0);
                if d > peak.1 {
                    peak = (sec / 3_600.0, d);
                }
                trough = trough.min(d);
            }
            (peak.0, peak.1 / trough)
        };

        let (greenwich, swing) = day(0.0);
        assert!(
            (greenwich - 14.5).abs() < 0.6,
            "the thermosphere peaks in the early afternoon, not at {greenwich} h UT"
        );
        assert!(
            swing > 2.0,
            "a day at 400 km is a factor of two or more in density, not {swing}"
        );

        // 60 degrees east is four hours earlier in universal time, exactly.
        let (east, swing_east) = day(60.0);
        assert!(
            (east - (greenwich - 4.0)).abs() < 0.6,
            "at 60 E the bulge is at {east} h UT where Greenwich has it at {greenwich} h"
        );
        assert!(swing_east > 2.0, "{swing_east}");
    }

    /// SDTM-1's own boundary, measured rather than inherited from the NRLMSISE-00 block above: this
    /// model is fitted here, so the wrapper's only job is the unit conversion, the derived local solar
    /// time and the refusal of a non-finite argument.
    #[test]
    fn the_sdtm1_profile_thins_with_height_and_refuses_a_non_finite_argument() {
        let at = |altitude_km: f64| {
            sl_sdtm1_density(172.0, 29_000.0, altitude_km, 0.0, 0.0, 150.0, 150.0, 0.0)
        };
        let (low, mid, high) = (at(200.0), at(400.0), at(600.0));
        assert!(low > mid && mid > high, "{low} {mid} {high}");
        // kg/m³: the model's own unit, so this pins the km → m conversion, not a g/cm³ factor.
        assert!((1e-13..1e-11).contains(&mid), "{mid}");

        // Every one of the eight arguments is refused when non-finite, not just the altitude. A NaN
        // day of year would otherwise reach only the seasonal harmonics and come back plausible.
        let base = [172.0, 29_000.0, 400.0, 0.0, 0.0, 150.0, 150.0, 0.0];
        for slot in 0..base.len() {
            let mut a = base;
            a[slot] = f64::NAN;
            assert!(
                sl_sdtm1_density(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]).is_nan(),
                "a NaN in slot {slot} answered a number"
            );
            assert!(
                sl_sdtm1_exospheric_temperature(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
                    .is_nan(),
                "a NaN in slot {slot} answered a temperature"
            );
        }
    }

    /// The point the wrapper assembles is four conversions the caller does not make: kilometres to
    /// metres, degrees to radians, a *fractional* day of year out of the day and the UT seconds, and a
    /// local solar time out of the UT seconds and the longitude. Each is asserted as a **value**,
    /// because the metamorphic form of this test — two calls a half-day and 180 degrees apart, which
    /// must name the same point and so must agree — is blind to the longitude term's scale *and* its
    /// sign at once: 360 degrees becomes 5 400 hours under `longitude · 15`, and 24 divides 5 400, so
    /// the pair still agrees. A symmetry cannot pin the constant the symmetry is built out of.
    #[test]
    fn the_sdtm1_point_is_the_callers_coordinates_in_the_fits_own() {
        // 15 h UT at 30 E is 17 h local: one hour of local time per fifteen degrees. Every quantity
        // here is exact in binary — 54 000/3 600, 30/15, and 54 000/86 400 = 5/8 — so these are
        // equalities and not tolerances in disguise.
        let p = sdtm1_point(172.0, 54_000.0, 400.0, 45.0, 30.0);
        assert_eq!(p.local_solar_time, 17.0);
        assert_eq!(p.altitude, 400_000.0);
        assert_eq!(p.latitude, 45.0_f64.to_radians());
        assert_eq!(p.day_of_year, 172.625);

        // Past the end of the day the local time wraps into the interval the diurnal basis is written
        // over, rather than leaning on the harmonics' periodicity: 23 h UT at 150 E is 33 h, so 9 h.
        assert_eq!(
            sdtm1_point(172.0, 82_800.0, 400.0, 0.0, 150.0).local_solar_time,
            9.0
        );
        // And from below: midnight UT at 180 W is −12 h, which is noon of the previous local day.
        assert_eq!(
            sdtm1_point(172.0, 0.0, 400.0, 0.0, -180.0).local_solar_time,
            12.0
        );

        // That point reaches the field, with the three drivers in the slots they are named for. The
        // values are distinct so that a swapped `f107`/`f107a` cannot survive: the reference builds
        // the record by name and the entry point builds it from its argument order.
        let reference = SDTM1_FIELD.density(
            &p,
            &SdtmDrivers {
                f107: 120.0,
                f107a: 160.0,
                storm: 30.0,
            },
        );
        let got = sl_sdtm1_density(172.0, 54_000.0, 400.0, 45.0, 30.0, 120.0, 160.0, 30.0);
        assert_eq!(got, reference, "{got} against {reference}");
    }

    /// The temperature the fit parameterises, and the driver it answers to. The range is the shipped
    /// field's own envelope over the six months it was fitted on; the storm direction is the sign the
    /// coefficients carry, which is the one thing a caller reading a `T∞` needs to be able to trust.
    #[test]
    fn the_sdtm1_exospheric_temperature_is_a_temperature_that_rises_with_the_storm() {
        let at = |storm: f64| {
            sl_sdtm1_exospheric_temperature(172.0, 29_000.0, 400.0, 0.0, 0.0, 150.0, 150.0, storm)
        };
        let quiet = at(0.0);
        assert!((586.0..1528.0).contains(&quiet), "{quiet} K");
        assert!(at(100.0) > quiet, "{} against {quiet}", at(100.0));
    }

    /// The shipped storm state cools, so a constant history answers its steady state
    /// `H/(1 + H/H_s)` from the first step, at any cadence — the field's own fixed point, which is
    /// the property worth checking across a language boundary.
    #[test]
    fn the_sdtm1_storm_state_holds_a_constant_history_at_its_steady_state() {
        let flat = [5.0_f64; 16];
        let spacelab_propagation::sdtm::StormResponse::Cooling(cooling) = SDTM1_FIELD.storm()
        else {
            panic!("the shipped field's storm state is the cooling one");
        };
        let steady = cooling.steady_state(5.0);
        assert!(steady < 5.0);
        for cadence in [10_800.0, 3_600.0] {
            let got = sdtm1_storm_response(&flat, cadence);
            assert!(
                (got - steady).abs() < 1e-12 * steady,
                "{got} at {cadence} s"
            );
        }
        // A cadence that is not a positive duration is not a history the state can read.
        assert!(sdtm1_storm_response(&flat, 0.0).is_nan());
        assert!(sdtm1_storm_response(&flat, -10_800.0).is_nan());
        assert!(sdtm1_storm_response(&flat, f64::NAN).is_nan());
        // A surge over a quiet background lands between the background and the peak.
        let mut surge = [5.0_f64; 16];
        surge[12..].fill(300.0);
        let response = sdtm1_storm_response(&surge, 10_800.0);
        assert!((steady..300.0).contains(&response), "{response}");

        // At the C boundary an empty array and a null pointer are different questions: `n = 0` is a
        // history with nothing in it, a null pointer is no history at all.
        assert_eq!(
            unsafe { sl_sdtm1_storm_response(flat.as_ptr(), 0, 10_800.0) },
            0.0
        );
        assert!(unsafe { sl_sdtm1_storm_response(core::ptr::null(), 16, 10_800.0) }.is_nan());
        assert_eq!(
            unsafe { sl_sdtm1_storm_response(flat.as_ptr(), flat.len(), 10_800.0) },
            sdtm1_storm_response(&flat, 10_800.0)
        );
    }

    /// The interval is log-normal about the density, so `lower · upper = density²` exactly whatever
    /// the regime and whatever `z` — the one property that holds across every one of the six fitted
    /// sigmas and therefore the one worth asserting at a boundary that cannot see the table.
    #[test]
    fn the_sdtm1_interval_is_log_normal_about_the_density_it_brackets() {
        let density = 4.118e-12;
        let mut out = [f64::MIN; SL_SDTM1_INTERVAL];
        let status = unsafe {
            sl_sdtm1_density_interval(
                density,
                150.0,
                150.0,
                0.0,
                1.644_853_626_951_472_7,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(status, SL_OK);
        let (lower, upper) = (out[0], out[1]);
        assert!(
            lower < density && density < upper,
            "{lower} {density} {upper}"
        );
        assert!(
            (lower * upper / (density * density) - 1.0).abs() < 1e-12,
            "{lower} x {upper} is not {density} squared"
        );

        // `z = 0` collapses the interval onto the density, which is the only value of `z` for which
        // the two ends coincide and so pins the multiplier's place in the exponent.
        let mut collapsed = [f64::MIN; SL_SDTM1_INTERVAL];
        assert_eq!(
            unsafe {
                sl_sdtm1_density_interval(density, 150.0, 150.0, 0.0, 0.0, collapsed.as_mut_ptr())
            },
            SL_OK
        );
        assert_eq!(collapsed, [density, density]);

        // The quiet low-flux cell is wider than the quiet high-flux one, so moving `f107a` across the
        // edges has to move the width.
        let mut quiet_sun = [f64::MIN; SL_SDTM1_INTERVAL];
        assert_eq!(
            unsafe {
                sl_sdtm1_density_interval(
                    density,
                    80.0,
                    80.0,
                    0.0,
                    1.644_853_626_951_472_7,
                    quiet_sun.as_mut_ptr(),
                )
            },
            SL_OK
        );
        assert!(
            quiet_sun[1] / quiet_sun[0] > upper / lower,
            "{:?} against {:?}",
            quiet_sun,
            out
        );

        // Moving the storm bin moves the interval too; the measured direction is not asserted, only
        // that the bin is read at all.
        let mut stormy = [f64::MIN; SL_SDTM1_INTERVAL];
        assert_eq!(
            unsafe {
                sl_sdtm1_density_interval(
                    density,
                    150.0,
                    150.0,
                    100.0,
                    1.644_853_626_951_472_7,
                    stormy.as_mut_ptr(),
                )
            },
            SL_OK
        );
        assert_ne!(stormy, out);

        assert_eq!(
            unsafe {
                sl_sdtm1_density_interval(density, 150.0, 150.0, 0.0, 1.0, core::ptr::null_mut())
            },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn the_analytical_propagator_starts_where_it_was_given_and_regresses_the_node() {
        let p = SlBrouwerLyddaneParams {
            initial: circular_state(7_000_000.0, 51.6_f64.to_radians()),
            mu: MU_EARTH,
            earth_radius_m: RE_EARTH,
            c20: -1.082_626_68e-3,
            c30: 2.532_656_49e-6,
            c40: 1.619_621_59e-6,
            c50: 2.272_960_83e-7,
            m2: 0.0,
        };
        let times = [0.0, 86_400.0];
        let mut out = [SlState {
            position: SlVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            velocity: SlVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        }; 2];
        assert_eq!(
            unsafe {
                sl_brouwer_lyddane_propagate(
                    &raw const p,
                    times.as_ptr(),
                    times.len(),
                    out.as_mut_ptr(),
                )
            },
            SL_OK
        );
        // The osculating→mean→osculating round trip at zero offset is the identity to the theory's
        // own truncation, which is metres and not kilometres.
        let d = [
            out[0].position.x - p.initial.position.x,
            out[0].position.y - p.initial.position.y,
            out[0].position.z - p.initial.position.z,
        ];
        let miss = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!(miss < 10.0, "the round trip missed by {miss} m");
        // A prograde orbit's node regresses under J2: about -5 deg/day at this altitude.
        let drift = (raan_of(out[1]) - raan_of(out[0])).to_degrees();
        assert!((-8.0..-3.0).contains(&drift), "{drift} deg/day");
    }

    #[test]
    fn a_propagation_the_theory_cannot_build_names_what_is_wrong_with_it() {
        let mut p = SlBrouwerLyddaneParams {
            initial: circular_state(7_000_000.0, 51.6_f64.to_radians()),
            mu: -1.0,
            earth_radius_m: RE_EARTH,
            c20: -1.082_626_68e-3,
            c30: 0.0,
            c40: 0.0,
            c50: 0.0,
            m2: 0.0,
        };
        let t = [0.0_f64];
        let mut out = [SlState {
            position: SlVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            velocity: SlVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        }; 1];
        assert_eq!(
            unsafe { sl_brouwer_lyddane_propagate(&raw const p, t.as_ptr(), 1, out.as_mut_ptr()) },
            SL_ERR_DOMAIN
        );
        p.mu = MU_EARTH;
        let bad = [f64::INFINITY];
        assert_eq!(
            unsafe {
                sl_brouwer_lyddane_propagate(&raw const p, bad.as_ptr(), 1, out.as_mut_ptr())
            },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn the_step2_corrections_are_periodic_in_sidereal_time() {
        let delaunay = [1.1, 2.2, 3.3, 4.4, 5.5];
        let mut a = [0.0_f64; 5];
        let mut b = [0.0_f64; 5];
        assert_eq!(
            unsafe { sl_solid_tide_step2(0.7, delaunay.as_ptr(), a.as_mut_ptr()) },
            SL_OK
        );
        assert_eq!(
            unsafe {
                sl_solid_tide_step2(
                    0.7 + 2.0 * core::f64::consts::PI,
                    delaunay.as_ptr(),
                    b.as_mut_ptr(),
                )
            },
            SL_OK
        );
        // The corrections are trigonometric in θ_f = m·(GMST + π) − Σ n_i F_i, so a full turn of
        // sidereal time is the identity up to the argument reduction's own rounding.
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-20, "{x} vs {y}");
        }
        assert!(a.iter().any(|v| v.abs() > 0.0), "every correction is zero");
        assert_eq!(
            unsafe { sl_solid_tide_step2(f64::NAN, delaunay.as_ptr(), a.as_mut_ptr()) },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn the_pole_tides_are_affine_in_the_wobble_and_the_two_poles_disagree() {
        let t = 20.0 * 365.25 * 86_400.0;
        let at = |xp: f64, tag: i32| {
            let mut out = [0.0_f64; 2];
            let status = unsafe { sl_solid_pole_tide(xp, 1e-7, t, tag, out.as_mut_ptr()) };
            assert_eq!(status, SL_OK);
            out
        };
        // IERS §6.4 is linear in (m1, m2), so a doubled offset gives a doubled difference.
        let (base, one, two) = (
            at(0.0, MEAN_POLE_SECULAR2018),
            at(1e-7, MEAN_POLE_SECULAR2018),
            at(2e-7, MEAN_POLE_SECULAR2018),
        );
        for k in 0..2 {
            let lhs = two[k] - base[k];
            let rhs = 2.0 * (one[k] - base[k]);
            // The two differences cancel about three digits, so the affinity is checked to the
            // precision that survives the cancellation rather than to the f64 floor.
            assert!((lhs - rhs).abs() <= 1e-9 * lhs.abs(), "{lhs} vs {rhs}");
        }
        // The retired pole is ~60 mas away, which is larger than the wobble the tide responds to —
        // the reason the tag exists at all rather than being a default nobody sees.
        let retired = at(0.0, MEAN_POLE_IERS2010);
        assert!(
            (retired[0] - base[0]).abs() > 0.1 * base[0].abs(),
            "{retired:?} vs {base:?}"
        );
        let mut ocean = [0.0_f64; 2];
        assert_eq!(
            unsafe { sl_ocean_pole_tide(1e-7, 1e-7, t, MEAN_POLE_SECULAR2018, ocean.as_mut_ptr()) },
            SL_OK
        );
        // The ocean pole tide is the smaller of the two on the same band, and not zero.
        assert!(ocean[0] != 0.0 && ocean[0].abs() < base[0].abs().max(1e-30) * 10.0);
        assert_eq!(
            unsafe { sl_solid_pole_tide(0.0, 0.0, t, 7, ocean.as_mut_ptr()) },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn the_conventional_zonals_are_one_fixed_mean_plus_the_secular_drift() {
        let zonal = |years: f64, f: unsafe extern "C" fn(f64, *mut SlZonal) -> i32| {
            let mut out = SlZonal {
                c20: 0.0,
                c30: 0.0,
                c40: 0.0,
            };
            assert_eq!(unsafe { f(years, &raw mut out) }, SL_OK);
            out
        };
        // conventional(t) − secular(t) is the epoch mean, so it does not move with t.
        let mean_at = |years: f64| {
            let c = zonal(years, sl_conventional_zonal_coefficients);
            let d = zonal(years, sl_time_variable_zonal_correction);
            [c.c20 - d.c20, c.c30 - d.c30, c.c40 - d.c40]
        };
        let (a, b) = (mean_at(0.0), mean_at(25.0));
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < 1e-24, "{} vs {}", a[k], b[k]);
        }
        // IERS 2010 Table 6.2 at J2000.0, and the tabulated dC20/dt over 25 years.
        assert!((a[0] - (-0.484_169_48e-3)).abs() < 1e-15, "{}", a[0]);
        let drift = zonal(25.0, sl_time_variable_zonal_correction);
        assert!((drift.c20 - 11.6e-12 * 25.0).abs() < 1e-24, "{}", drift.c20);
        let mut out = SlZonal {
            c20: 0.0,
            c30: 0.0,
            c40: 0.0,
        };
        assert_eq!(
            unsafe { sl_conventional_zonal_coefficients(f64::NAN, &raw mut out) },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn the_panel_coefficients_and_the_speed_ratio_are_the_scalings_they_promise() {
        let s = sl_molecular_speed_ratio(7600.0, 1000.0, 0.016);
        assert!((s - 7.4549).abs() < 1e-3, "{s}");
        // s = v·sqrt(M / 2RT): linear in the speed, and inverse in the square root of temperature.
        assert!(
            (sl_molecular_speed_ratio(15_200.0, 1000.0, 0.016) - 2.0 * s).abs() < 1e-12 * s,
            "the speed ratio is not linear in the speed"
        );
        assert!(
            (sl_molecular_speed_ratio(7600.0, 4000.0, 0.016) - 0.5 * s).abs() < 1e-12 * s,
            "the speed ratio does not scale as 1/sqrt(T)"
        );
        let mut panel = SlDragPanel {
            pressure: 0.0,
            shear: 0.0,
            drag: 0.0,
            lift: 0.0,
        };
        assert_eq!(
            unsafe { sl_sentman_drag_coefficients(s, 1.0, 1.0, 300.0, 1000.0, &raw mut panel) },
            SL_OK
        );
        // Head-on there is no tangential component to shear against, and the drag sits where every
        // free-molecular treatment puts it for a fully accommodating surface.
        assert!(panel.shear.abs() < 1e-12, "{}", panel.shear);
        assert!((panel.drag - 2.148_22).abs() < 1e-4, "{}", panel.drag);
        assert_eq!(
            unsafe { sl_sentman_drag_coefficients(s, 1.0, 1.0, 0.0, 1000.0, &raw mut panel) },
            SL_ERR_DOMAIN
        );
    }

    #[test]
    fn the_density_correction_recovers_the_field_it_was_fitted_to() {
        // A field the model can represent exactly: constant plus one harmonic in each variable.
        let truth = |lat: f64, lst: f64| {
            0.05 + 0.02 * lat.sin() + 0.03 * (core::f64::consts::PI * lst / 12.0).cos()
        };
        let samples: Vec<SlDensitySample> = (0..48)
            .map(|i| {
                let lat = -1.2 + 0.05 * f64::from(i);
                let lst = f64::from(i) * 0.5;
                SlDensitySample {
                    latitude_rad: lat,
                    local_solar_time_h: lst,
                    log_ratio: truth(lat, lst),
                    weight: 1.0,
                }
            })
            .collect();
        let coefficients = collect(0.0_f64, |out, cap, len| unsafe {
            sl_density_correction_fit(samples.as_ptr(), samples.len(), 1e-12, out, cap, len)
        })
        .expect("the fit");
        assert_eq!(coefficients.len(), DENSITY_CORRECTION_COEFFICIENTS);
        for s in samples.iter().step_by(7) {
            let mut factor = 0.0;
            assert_eq!(
                unsafe {
                    sl_density_correction_factor(
                        coefficients.as_ptr(),
                        coefficients.len(),
                        s.latitude_rad,
                        s.local_solar_time_h,
                        &raw mut factor,
                    )
                },
                SL_OK
            );
            // The factor is the exponential of the log-ratio the fit reproduces.
            assert!(
                (factor - s.log_ratio.exp()).abs() < 1e-6,
                "{factor} vs {}",
                s.log_ratio.exp()
            );
        }
        // A coefficient vector of the wrong length is a different model, not a truncated one.
        let mut factor = 0.0;
        assert_eq!(
            unsafe {
                sl_density_correction_factor(coefficients.as_ptr(), 4, 0.0, 12.0, &raw mut factor)
            },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn the_density_grid_puts_each_sample_in_the_cell_it_names() {
        let samples = [
            SlDensitySample {
                latitude_rad: -1.5,
                local_solar_time_h: 1.0,
                log_ratio: 0.25,
                weight: 1.0,
            },
            SlDensitySample {
                latitude_rad: 1.5,
                local_solar_time_h: 23.0,
                log_ratio: -0.5,
                weight: 1.0,
            },
        ];
        let grid = collect(0.0_f64, |out, cap, len| unsafe {
            sl_density_grid_snapshot(samples.as_ptr(), samples.len(), 2, 2, out, cap, len)
        })
        .expect("the grid");
        assert_eq!(grid.len(), 4);
        // Row-major `lat_bin * n_lst + lst_bin`: the southern morning cell and the northern night
        // one, and the two the samples never reach are exactly zero rather than interpolated into.
        assert!((grid[0] - 0.25).abs() < 1e-15, "{grid:?}");
        assert!((grid[3] - (-0.5)).abs() < 1e-15, "{grid:?}");
        assert!(grid[1] == 0.0 && grid[2] == 0.0, "{grid:?}");
        let mut len = 0_usize;
        assert_eq!(
            unsafe {
                sl_density_grid_snapshot(
                    samples.as_ptr(),
                    samples.len(),
                    0,
                    2,
                    core::ptr::null_mut(),
                    0,
                    &raw mut len,
                )
            },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn the_space_weather_file_answers_the_days_it_holds() {
        let days = collect(
            SlSpaceWeatherDay {
                year: 0,
                month: 0,
                day: 0,
                ap: 0.0,
                f107_observed: 0.0,
                f107_center81: 0.0,
            },
            |out, cap, len| unsafe {
                sl_space_weather_days(SW_CSV.as_ptr().cast(), SW_CSV.len(), out, cap, len)
            },
        )
        .expect("the days");
        assert_eq!(days.len(), 7);
        let d = days
            .iter()
            .find(|d| (d.year, d.month, d.day) == (1957, 10, 1))
            .expect("1957-10-01");
        assert!((d.ap - 21.0).abs() < 1e-9 && (d.f107_observed - 269.3).abs() < 1e-9);
        // The look-back the array needs runs past the start of a seven-day file, so the last day is
        // the one that can be answered and the first is not.
        let mut ap = [0.0_f64; AP_ARRAY_LEN];
        let status = unsafe {
            sl_space_weather_ap_array(
                SW_CSV.as_ptr().cast(),
                SW_CSV.len(),
                1957,
                10,
                1,
                0.0,
                ap.as_mut_ptr(),
            )
        };
        assert_eq!(status, SL_ERR_DOMAIN);
        let mut drivers = [0.0_f64; DENSITY_DRIVERS];
        assert_eq!(
            unsafe { sl_space_weather_drivers(150.0, 130.0, 9.0, drivers.as_mut_ptr()) },
            SL_OK
        );
        // [flux, ap, flux², √ap, 1] on the mean of the two fluxes.
        assert!((drivers[0] - 140.0).abs() < 1e-12, "{drivers:?}");
        assert!((drivers[2] - 140.0 * 140.0).abs() < 1e-9, "{drivers:?}");
        assert!((drivers[3] - 3.0).abs() < 1e-12, "{drivers:?}");
        assert!((drivers[4] - 1.0).abs() < 1e-15, "{drivers:?}");
    }

    #[test]
    fn the_density_file_reduces_to_the_observations_it_holds() {
        let samples = collect(
            SlDensitySample {
                latitude_rad: 0.0,
                local_solar_time_h: 0.0,
                log_ratio: 0.0,
                weight: 0.0,
            },
            |out, cap, len| unsafe {
                sl_density_data_samples(CHAMP.as_ptr().cast(), CHAMP.len(), out, cap, len)
            },
        )
        .expect("the samples");
        assert!(!samples.is_empty());
        assert!(
            samples
                .iter()
                .all(|s| s.log_ratio.is_finite() && s.weight > 0.0)
        );
        let normalized = |reference: f64| {
            collect(
                SlDensityObservation {
                    year: 0,
                    month: 0,
                    day: 0,
                    latitude_rad: 0.0,
                    local_solar_time_h: 0.0,
                    ln_density_ref: 0.0,
                },
                |out, cap, len| unsafe {
                    sl_density_altitude_normalized(
                        CHAMP.as_ptr().cast(),
                        CHAMP.len(),
                        reference,
                        60_000.0,
                        out,
                        cap,
                        len,
                    )
                },
            )
            .expect("the observations")
        };
        let (a, b) = (normalized(450_000.0), normalized(390_000.0));
        assert_eq!(a.len(), 30);
        assert_eq!(a.len(), b.len());
        // Lowering the reference by exactly one scale height adds exactly ln e to every log-density.
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!((x.year, x.month, x.day), (y.year, y.month, y.day));
            assert!((y.ln_density_ref - x.ln_density_ref - 1.0).abs() < 1e-12);
        }
        let mut len = 0_usize;
        assert_eq!(
            unsafe {
                sl_density_altitude_normalized(
                    CHAMP.as_ptr().cast(),
                    CHAMP.len(),
                    450_000.0,
                    0.0,
                    core::ptr::null_mut(),
                    0,
                    &raw mut len,
                )
            },
            SL_ERR_DOMAIN
        );
    }

    #[test]
    fn the_swath_closes_the_triangle_it_describes() {
        let mut g = SlSwath {
            earth_central_angle_rad: 0.0,
            elevation_rad: 0.0,
            ground_range_m: 0.0,
            slant_range_m: 0.0,
        };
        let (re, altitude, look) = (RE_EARTH, 700_000.0, 0.35);
        assert_eq!(
            unsafe { sl_swath_geometry(re, altitude, look, &raw mut g) },
            SL_OK
        );
        // The three angles of the sub-satellite/target/centre triangle close on π.
        let sum =
            g.earth_central_angle_rad + look + (core::f64::consts::FRAC_PI_2 + g.elevation_rad);
        assert!(
            (sum - core::f64::consts::PI).abs() < 1e-12,
            "the triangle sums to {sum}"
        );
        // Ground range is the arc the central angle subtends, and the slant range closes the same
        // triangle by the law of sines.
        assert!((g.ground_range_m - re * g.earth_central_angle_rad).abs() < 1e-6);
        let expected = re * g.earth_central_angle_rad.sin() / look.sin();
        assert!(
            (g.slant_range_m - expected).abs() < 1e-6 * expected,
            "{} vs {expected}",
            g.slant_range_m
        );
        // Past the horizon the line of sight misses the Earth, and there is no answer to round.
        assert_eq!(
            unsafe { sl_swath_geometry(re, altitude, 1.4, &raw mut g) },
            SL_ERR_DOMAIN
        );
        assert_eq!(
            unsafe { sl_swath_geometry(re, -1.0, look, &raw mut g) },
            SL_ERR_DOMAIN
        );
    }

    /// A fixed set of Earth-orientation quantities: a real epoch and representative polar motion,
    /// so the rotation the propagators build is the one an operational arc would use.
    static EOP: SlEop = SlEop {
        epoch_tt_ns: 725_760_069_184_000_000,
        epoch_ut1_ns: 725_760_000_000_000_000,
        xp_rad: 1.0e-6,
        yp_rad: 2.0e-6,
        dx_rad: 0.0,
        dy_rad: 0.0,
    };

    /// Build an ephemeris, read it at `t`, and close it — so no test below can leak a handle by
    /// forgetting the third step.
    fn sample_ephemeris(
        build: impl FnOnce(*mut *mut SlEphemeris) -> i32,
        t: f64,
    ) -> Result<SlState, i32> {
        let mut handle: *mut SlEphemeris = core::ptr::null_mut();
        let status = build(&raw mut handle);
        if status != SL_OK {
            return Err(status);
        }
        let mut out = SlState {
            position: SlVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            velocity: SlVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let status = unsafe { sl_ephemeris_at(handle, t, &raw mut out) };
        unsafe { sl_ephemeris_close(handle) };
        if status == SL_OK {
            Ok(out)
        } else {
            Err(status)
        }
    }

    fn miss(a: SlState, b: SlState) -> f64 {
        let d = [
            a.position.x - b.position.x,
            a.position.y - b.position.y,
            a.position.z - b.position.z,
        ];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    #[test]
    fn the_two_integrators_agree_on_the_forces_they_share() {
        let s0 = circular_state(7_000_000.0, 51.6_f64.to_radians());
        let arc = 600.0;
        let ias15 = sample_ephemeris(
            |out| unsafe {
                sl_ephemeris_j2(&raw const s0, arc, MU_EARTH, 1.082_626_68e-3, RE_EARTH, out)
            },
            arc,
        )
        .expect("IAS15");
        let dop853 = sample_ephemeris(
            |out| unsafe {
                sl_ephemeris_j2_dense(
                    &raw const s0,
                    arc,
                    MU_EARTH,
                    1.082_626_68e-3,
                    RE_EARTH,
                    1e-12,
                    out,
                )
            },
            arc,
        )
        .expect("DOP853");
        // Two independent methods on identical physics: the difference is each one's truncation,
        // and at these tolerances that is millimetres over ten minutes, not metres.
        assert!(miss(ias15, dop853) < 1e-3, "{} m", miss(ias15, dop853));
    }

    #[test]
    fn a_vacuum_atmosphere_is_the_geopotential_it_was_added_to() {
        let s0 = circular_state(7_000_000.0, 51.6_f64.to_radians());
        let arc = 600.0;
        let bare = sample_ephemeris(
            |out| unsafe { sl_ephemeris_geopotential(&raw const s0, arc, 4, 0.0, 1e-11, out) },
            arc,
        )
        .expect("geopotential");
        let vacuum = SlExponentialAtmosphere {
            ballistic_coefficient: 100.0,
            reference_density: 0.0,
            reference_altitude_m: 500_000.0,
            scale_height_m: 60_000.0,
            earth_radius_m: RE_EARTH,
        };
        let with_drag = sample_ephemeris(
            |out| unsafe {
                sl_ephemeris_drag(&raw const s0, arc, 4, 0.0, &raw const vacuum, 1e-11, out)
            },
            arc,
        )
        .expect("drag");
        // A zero reference density is a vacuum, and a vacuum exerts no drag — so the two arcs are
        // the same integration and differ only by the step control seeing an extra zero term.
        assert!(miss(bare, with_drag) < 1e-6, "{} m", miss(bare, with_drag));
        // A real atmosphere does something, and what it does is take energy out.
        let thick = SlExponentialAtmosphere {
            reference_density: 1e-11,
            ..vacuum
        };
        let dragged = sample_ephemeris(
            |out| unsafe {
                sl_ephemeris_drag(&raw const s0, arc, 4, 0.0, &raw const thick, 1e-11, out)
            },
            arc,
        )
        .expect("drag");
        assert!(miss(bare, dragged) > 1e-6, "{} m", miss(bare, dragged));
        let bad = SlExponentialAtmosphere {
            scale_height_m: 0.0,
            ..vacuum
        };
        assert!(matches!(
            sample_ephemeris(
                |out| unsafe {
                    sl_ephemeris_drag(&raw const s0, arc, 4, 0.0, &raw const bad, 1e-11, out)
                },
                arc,
            ),
            Err(SL_ERR_DOMAIN)
        ));
    }

    #[test]
    fn a_constant_orientation_series_is_the_fixed_orientation_it_repeats() {
        let s0 = circular_state(7_000_000.0, 51.6_f64.to_radians());
        let arc = 600.0;
        let fixed = sample_ephemeris(
            |out| unsafe {
                sl_ephemeris_j2_itrf(
                    &raw const s0,
                    arc,
                    MU_EARTH,
                    1.082_626_68e-3,
                    RE_EARTH,
                    &raw const EOP,
                    1e-11,
                    out,
                )
            },
            arc,
        )
        .expect("fixed EOP");
        // The same orientation written as a two-row series that never changes. The fixed form grids
        // and interpolates the CIP; the series form calls the full chain at every stage — so this
        // measures the interpolation error, which the grid is built to keep sub-millimetre.
        let row = |t_seconds: f64| SlEopSample {
            t_seconds,
            xp_rad: EOP.xp_rad,
            yp_rad: EOP.yp_rad,
            dut1_seconds: (EOP.epoch_ut1_ns - EOP.epoch_tt_ns) as f64 / 1e9,
            dx_rad: EOP.dx_rad,
            dy_rad: EOP.dy_rad,
        };
        let series = [row(-arc), row(2.0 * arc)];
        let interpolated = sample_ephemeris(
            |out| unsafe {
                sl_ephemeris_j2_itrf_series(
                    &raw const s0,
                    arc,
                    MU_EARTH,
                    1.082_626_68e-3,
                    RE_EARTH,
                    EOP.epoch_tt_ns,
                    0.0,
                    series.as_ptr(),
                    series.len(),
                    1e-11,
                    out,
                )
            },
            arc,
        )
        .expect("EOP series");
        assert!(
            miss(fixed, interpolated) < 1e-3,
            "{} m",
            miss(fixed, interpolated)
        );
        // One row cannot be interpolated, and a series that goes backwards is not a series.
        for bad in [
            series[..1].to_vec(),
            vec![row(10.0), row(-10.0)],
            vec![row(0.0), row(0.0)],
        ] {
            assert!(matches!(
                sample_ephemeris(
                    |out| unsafe {
                        sl_ephemeris_j2_itrf_series(
                            &raw const s0,
                            arc,
                            MU_EARTH,
                            1.082_626_68e-3,
                            RE_EARTH,
                            EOP.epoch_tt_ns,
                            0.0,
                            bad.as_ptr(),
                            bad.len(),
                            1e-11,
                            out,
                        )
                    },
                    arc,
                ),
                Err(SL_ERR_INVALID_ARGUMENT)
            ));
        }
        // Every field of every row has to be finite, not just the one the ordering is defined on: a
        // NaN in the payload reaches the rotation through `lerp_at`, and the step controller then
        // rejects every step until it runs out of them. The row index is part of the refusal, so a
        // caller with a thousand rows is told which one.
        for (index, field) in [
            (1, "t_seconds"),
            (0, "xp_rad"),
            (0, "yp_rad"),
            (1, "dut1_seconds"),
            (0, "dx_rad"),
            (1, "dy_rad"),
        ] {
            let mut bad = series.to_vec();
            let poisoned = &mut bad[index];
            match field {
                "t_seconds" => poisoned.t_seconds = f64::NAN,
                "xp_rad" => poisoned.xp_rad = f64::INFINITY,
                "yp_rad" => poisoned.yp_rad = f64::NAN,
                "dut1_seconds" => poisoned.dut1_seconds = f64::NAN,
                "dx_rad" => poisoned.dx_rad = f64::NAN,
                _ => poisoned.dy_rad = f64::NEG_INFINITY,
            }
            assert!(matches!(
                sample_ephemeris(
                    |out| unsafe {
                        sl_ephemeris_j2_itrf_series(
                            &raw const s0,
                            arc,
                            MU_EARTH,
                            1.082_626_68e-3,
                            RE_EARTH,
                            EOP.epoch_tt_ns,
                            0.0,
                            bad.as_ptr(),
                            bad.len(),
                            1e-11,
                            out,
                        )
                    },
                    arc,
                ),
                Err(SL_ERR_INVALID_ARGUMENT)
            ));
            let message = last_error();
            assert!(
                message.contains(&format!("row {index} ")) && message.contains(field),
                "{message}"
            );
        }
    }

    #[test]
    fn every_added_force_moves_the_orbit_by_the_amount_it_is_worth() {
        let s0 = circular_state(7_000_000.0, 51.6_f64.to_radians());
        let arc = 600.0;
        let itrf = sample_ephemeris(
            |out| unsafe {
                sl_ephemeris_geopotential_itrf(&raw const s0, arc, 4, &raw const EOP, 1e-11, out)
            },
            arc,
        )
        .expect("ITRF geopotential");
        let tided = sample_ephemeris(
            |out| unsafe {
                sl_ephemeris_geopotential_tide(
                    &raw const s0,
                    arc,
                    4,
                    &raw const EOP,
                    0.0,
                    MU_SUN,
                    MU_MOON,
                    1e-11,
                    out,
                )
            },
            arc,
        )
        .expect("tide");
        // The degree-2 solid tide is a ~1e-8 m/s² perturbation, so over ten minutes it is sub-metre
        // and not sub-nanometre. Bounding it from both sides is what makes this a test rather than
        // an assertion that something happened.
        let tide = miss(itrf, tided);
        assert!((1e-9..1.0).contains(&tide), "the tide moved it {tide} m");

        let two_body = sample_ephemeris(
            |out| unsafe {
                sl_ephemeris_j2_dense(&raw const s0, arc, MU_EARTH, 0.0, RE_EARTH, 1e-12, out)
            },
            arc,
        )
        .expect("two-body");
        let relativistic = sample_ephemeris(
            |out| unsafe {
                sl_ephemeris_relativistic(&raw const s0, arc, 0.0, MU_EARTH, MU_SUN, 1e-12, out)
            },
            arc,
        )
        .expect("relativistic");
        // Schwarzschild dominates at ~1e-9 m/s²; over ten minutes that is well under a millimetre.
        let relativity = miss(two_body, relativistic);
        assert!(
            (1e-9..1e-2).contains(&relativity),
            "relativity moved it {relativity} m"
        );

        let bare = sample_ephemeris(
            |out| unsafe { sl_ephemeris_geopotential(&raw const s0, arc, 4, 0.0, 1e-11, out) },
            arc,
        )
        .expect("geopotential");
        let luni_solar = sample_ephemeris(
            |out| unsafe {
                sl_ephemeris_luni_solar(&raw const s0, arc, 4, 0.0, 1e-7, 0.0, 1e-11, out)
            },
            arc,
        )
        .expect("luni-solar");
        assert!(miss(bare, luni_solar) > 1e-9);

        let drag = SlNrlmsiseDrag {
            doy: 172,
            seconds_of_day: 29_000.0,
            ballistic_coefficient: 100.0,
            f107: 150.0,
            f107a: 150.0,
            ap: 4.0,
            mu: MU_EARTH,
        };
        let thermosphere = sample_ephemeris(
            |out| unsafe {
                sl_ephemeris_drag_nrlmsise00(
                    &raw const s0,
                    arc,
                    &raw const EOP,
                    &raw const drag,
                    1e-10,
                    out,
                )
            },
            arc,
        )
        .expect("NRLMSISE-00 drag");
        let kepler = sample_ephemeris(
            |out| unsafe {
                sl_ephemeris_j2_dense(&raw const s0, arc, MU_EARTH, 0.0, RE_EARTH, 1e-10, out)
            },
            arc,
        )
        .expect("two-body");
        assert!(miss(kepler, thermosphere) > 1e-9);
    }

    #[test]
    fn a_dense_constructor_refuses_an_arc_it_cannot_integrate() {
        let s0 = circular_state(7_000_000.0, 0.0);
        let mut handle: *mut SlEphemeris = core::ptr::null_mut();
        let at = |t1: f64, tolerance: f64, degree: u32, handle: *mut *mut SlEphemeris| unsafe {
            sl_ephemeris_geopotential_itrf(
                &raw const s0,
                t1,
                degree,
                &raw const EOP,
                tolerance,
                handle,
            )
        };
        assert_eq!(at(0.0, 1e-9, 4, &raw mut handle), SL_ERR_INVALID_ARGUMENT);
        assert_eq!(
            at(f64::INFINITY, 1e-9, 4, &raw mut handle),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(at(600.0, 0.0, 4, &raw mut handle), SL_ERR_INVALID_ARGUMENT);
        assert_eq!(
            at(600.0, 1e-9, MODEL_MAX_DEGREE as u32 + 1, &raw mut handle),
            SL_ERR_DOMAIN
        );
        // A refused constructor writes nothing, so the caller's pointer is still what it was.
        assert!(handle.is_null());
        assert_eq!(
            unsafe {
                sl_ephemeris_geopotential_itrf(
                    &raw const s0,
                    600.0,
                    4,
                    core::ptr::null(),
                    1e-9,
                    &raw mut handle,
                )
            },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn the_two_arithmetics_answer_the_same_orbit_to_the_precision_they_carry() {
        let s0 = circular_state(7_000_000.0, 51.6_f64.to_radians());
        let at = |precision: i32| {
            let mut out = SlState {
                position: SlVec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                velocity: SlVec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            };
            let status = unsafe {
                sl_propagate_j2_precision(
                    &raw const s0,
                    MU_EARTH,
                    1.082_626_68e-3,
                    RE_EARTH,
                    1.0,
                    600,
                    precision,
                    &raw mut out,
                )
            };
            assert_eq!(status, SL_OK);
            out
        };
        let (ordinary, extended) = (at(PRECISION_DOUBLE), at(PRECISION_DOUBLE_DOUBLE));
        // Same integrator, same steps, two arithmetics: the gap is the round-off the `f64` answer
        // carries, which over 600 steps is microns and not metres.
        let gap = miss(ordinary, extended);
        assert!(gap < 1e-3, "the two arithmetics differ by {gap} m");
        assert!(gap > 0.0, "double-double reproduced f64 bit for bit");
        let mut out = SlState {
            position: SlVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            velocity: SlVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        assert_eq!(
            unsafe {
                sl_propagate_j2_precision(
                    &raw const s0,
                    MU_EARTH,
                    0.0,
                    RE_EARTH,
                    1.0,
                    10,
                    7,
                    &raw mut out,
                )
            },
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe {
                sl_propagate_j2_precision(
                    &raw const s0,
                    MU_EARTH,
                    0.0,
                    RE_EARTH,
                    f64::NAN,
                    10,
                    PRECISION_DOUBLE,
                    &raw mut out,
                )
            },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    /// The elevation steering vector the tomographic estimators are matched to, written out here
    /// rather than borrowed from the crate so that the test states the convention it checks.
    fn steering(
        baselines: &[f64],
        wavelength: f64,
        slant_range: f64,
        elevation: f64,
    ) -> Vec<SlComplex> {
        baselines
            .iter()
            .map(|&b| {
                let phase =
                    -4.0 * core::f64::consts::PI * b * elevation / (wavelength * slant_range);
                SlComplex {
                    re: phase.cos(),
                    im: phase.sin(),
                }
            })
            .collect()
    }

    /// The index of the largest entry — which elevation an estimator says the scatterer is at.
    fn argmax(values: &[f64]) -> usize {
        let mut best = 0;
        for (i, v) in values.iter().enumerate() {
            if v > &values[best] {
                best = i;
            }
        }
        best
    }

    /// The tomographic geometry every spectral test below is run on: eight irregular baselines,
    /// X-band, a 700 km slant range, and an elevation grid that contains the injected scatterer.
    fn tomography() -> (Vec<f64>, Vec<f64>, SlTomoParams) {
        let baselines = vec![-160.0, -110.0, -70.0, -30.0, 20.0, 65.0, 120.0, 175.0];
        let elevations: Vec<f64> = (-20..=20).map(|k| f64::from(k) * 2.0).collect();
        let params = SlTomoParams {
            wavelength: 0.031,
            slant_range: 700_000.0,
            loading: 0.1,
            lambda: 0.05,
            max_iter: 600,
            threshold_rel: 0.5,
        };
        (baselines, elevations, params)
    }

    #[test]
    fn the_wrap_is_a_modular_reduction_and_not_a_clamp() {
        // A clamp would agree with a reduction on the principal branch and disagree everywhere
        // else, so the branch shift is the whole property: adding a whole turn must change
        // nothing.
        for step in -30..=30 {
            let phi = f64::from(step) * 0.21;
            let reduced = sl_wrap_phase(phi);
            assert!(
                reduced > -core::f64::consts::PI && reduced <= core::f64::consts::PI + 1e-12,
                "{phi} reduced to {reduced}, outside the principal branch"
            );
            assert!(
                (sl_wrap_phase(reduced) - reduced).abs() < 1e-12,
                "the reduction is not idempotent at {phi}"
            );
            for turns in -3..=3 {
                let shifted = sl_wrap_phase(phi + f64::from(turns) * core::f64::consts::TAU);
                assert!(
                    (shifted - reduced).abs() < 1e-9,
                    "{turns} turns moved the reduction of {phi} from {reduced} to {shifted}"
                );
            }
        }
    }

    #[test]
    fn a_chirp_is_all_phase_and_compresses_against_itself_to_its_own_energy() {
        let n = 64;
        let chirp = collect(SlComplex { re: 0.0, im: 0.0 }, |out, cap, len| unsafe {
            sl_lfm_chirp(n, 5.0e11, 3.0e7, out, cap, len)
        })
        .expect("the chirp is well posed");
        assert_eq!(chirp.len(), n);
        for c in &chirp {
            let modulus = c.re.hypot(c.im);
            assert!(
                (modulus - 1.0).abs() < 1e-12,
                "a frequency modulation changed the amplitude: |c| = {modulus}"
            );
        }

        // Matched filtering is a correlation, so the peak of a signal against itself is its own
        // energy -- n, for n unit-modulus samples -- and nothing in the output can exceed it.
        let compressed = collect(SlComplex { re: 0.0, im: 0.0 }, |out, cap, len| unsafe {
            sl_matched_filter(
                chirp.as_ptr(),
                chirp.len(),
                chirp.as_ptr(),
                chirp.len(),
                out,
                cap,
                len,
            )
        })
        .expect("the compression is well posed");
        let peak = compressed
            .iter()
            .map(|c| c.re.hypot(c.im))
            .fold(0.0_f64, f64::max);
        #[allow(clippy::cast_precision_loss)] // 64 is exact in f64
        let energy = n as f64;
        assert!(
            (peak - energy).abs() < 1e-6 * energy,
            "the compressed peak is {peak}, not the {energy} of energy that went in"
        );
    }

    #[test]
    fn an_image_against_itself_has_no_phase_and_perfect_coherence() {
        let (rows, cols) = (8_usize, 8_usize);
        let image: Vec<SlComplex> = (0..rows * cols)
            .map(|k| {
                #[allow(clippy::cast_precision_loss)] // 64 samples
                let phase = k as f64 * 0.37;
                SlComplex {
                    re: 2.0 * phase.cos(),
                    im: 2.0 * phase.sin(),
                }
            })
            .collect();

        let interferogram = collect(SlComplex { re: 0.0, im: 0.0 }, |out, cap, len| unsafe {
            sl_interferogram(image.as_ptr(), image.as_ptr(), image.len(), out, cap, len)
        })
        .expect("the pair is well posed");
        for c in &interferogram {
            assert!(
                c.im.abs() < 1e-12 && c.re > 0.0,
                "an image interfered with itself is not real-positive: {c:?}"
            );
        }

        let coherence = collect(0.0_f64, |out, cap, len| unsafe {
            sl_coherence(image.as_ptr(), image.as_ptr(), rows, cols, 1, out, cap, len)
        })
        .expect("the pair is well posed");
        assert_eq!(coherence.len(), rows * cols);
        for g in &coherence {
            assert!(
                (g - 1.0).abs() < 1e-12,
                "an image is not perfectly coherent with itself: {g}"
            );
        }
    }

    #[test]
    fn a_ramp_survives_wrapping_and_the_three_unwrappers_agree_on_it() {
        // A gradient under pi per sample is exactly the case every unwrapper is supposed to get
        // right, and it is the only case where they are required to agree: the raster has no
        // residues, so there is nothing for them to disagree about.
        let (rows, cols) = (10_usize, 12_usize);
        let mut truth = vec![0.0; rows * cols];
        let mut wrapped = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                #[allow(clippy::cast_precision_loss)] // a 10x12 raster
                let phase = 0.30 * r as f64 + 0.22 * c as f64;
                truth[r * cols + c] = phase;
                wrapped[r * cols + c] = sl_wrap_phase(phase);
            }
        }

        let charges = collect(0_i8, |out, cap, len| unsafe {
            sl_residues(wrapped.as_ptr(), rows, cols, out, cap, len)
        })
        .expect("the raster is well posed");
        assert_eq!(charges.len(), (rows - 1) * (cols - 1));
        assert!(
            charges.iter().all(|&q| q == 0),
            "a linear ramp cannot have a residue, and this one has {}",
            charges.iter().filter(|&&q| q != 0).count()
        );

        type Unwrapper =
            unsafe extern "C" fn(*const f64, usize, usize, *mut f64, usize, *mut usize) -> i32;
        for (name, unwrap) in [
            ("flood fill", sl_unwrap_phase_2d as Unwrapper),
            ("branch cut", sl_unwrap_phase_2d_branch_cut as Unwrapper),
            ("least squares", sl_unwrap_phase_2d_lstsq as Unwrapper),
        ] {
            let recovered = collect(0.0_f64, |out, cap, len| unsafe {
                unwrap(wrapped.as_ptr(), rows, cols, out, cap, len)
            })
            .expect("the raster is well posed");
            assert_eq!(recovered.len(), rows * cols);
            // Unwrapping fixes the phase only up to a whole-turn constant, so the property is
            // that the offset is *the same everywhere*, not that it is zero.
            let offsets: Vec<f64> = recovered
                .iter()
                .zip(&truth)
                .map(|(got, want)| got - want)
                .collect();
            let mean = offsets.iter().sum::<f64>() / offsets.len() as f64;
            let worst = offsets
                .iter()
                .map(|o| (o - mean).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                worst < 1e-9,
                "{name} did not recover the ramp: the offset varies by {worst} rad across it"
            );
        }
    }

    #[test]
    fn the_one_dimensional_unwrapper_recovers_the_profile_it_was_given() {
        let truth: Vec<f64> = (0..40).map(|k| f64::from(k) * 0.4).collect();
        let wrapped: Vec<f64> = truth.iter().map(|&p| sl_wrap_phase(p)).collect();
        let recovered = collect(0.0_f64, |out, cap, len| unsafe {
            sl_unwrap_phase_1d(wrapped.as_ptr(), wrapped.len(), out, cap, len)
        })
        .expect("the profile is well posed");
        for (got, want) in recovered.iter().zip(&truth) {
            assert!(
                (got - want - (recovered[0] - truth[0])).abs() < 1e-12,
                "the unwrapped profile drifts from the one it was built from"
            );
        }
    }

    #[test]
    fn both_elevation_spectra_peak_where_the_scatterer_actually_is() {
        let (baselines, elevations, params) = tomography();
        let truth = 10.0;
        let steer = steering(&baselines, params.wavelength, params.slant_range, truth);
        // One scatterer is a rank-one covariance: R = a a^H. Capon needs the diagonal loaded to
        // invert it, which is what `loading` is for; Bartlett does not invert anything.
        let mut covariance = vec![SlComplex { re: 0.0, im: 0.0 }; steer.len() * steer.len()];
        for i in 0..steer.len() {
            for j in 0..steer.len() {
                covariance[i * steer.len() + j] = SlComplex {
                    re: steer[i].re * steer[j].re + steer[i].im * steer[j].im,
                    im: steer[i].im * steer[j].re - steer[i].re * steer[j].im,
                };
            }
        }

        type Spectrum = unsafe extern "C" fn(
            *const SlComplex,
            *const f64,
            usize,
            *const f64,
            usize,
            *const SlTomoParams,
            *mut f64,
            usize,
            *mut usize,
        ) -> i32;
        for (name, spectrum) in [
            ("Capon", sl_capon_spectrum as Spectrum),
            ("Bartlett", sl_bartlett_spectrum as Spectrum),
        ] {
            let power = collect(0.0_f64, |out, cap, len| unsafe {
                spectrum(
                    covariance.as_ptr(),
                    baselines.as_ptr(),
                    baselines.len(),
                    elevations.as_ptr(),
                    elevations.len(),
                    &raw const params,
                    out,
                    cap,
                    len,
                )
            })
            .expect("the geometry is well posed");
            assert_eq!(power.len(), elevations.len());
            let found = elevations[argmax(&power)];
            assert!(
                (found - truth).abs() < 1e-9,
                "{name} put the scatterer at {found} m, not the {truth} m it was given"
            );
        }
    }

    #[test]
    fn sl1mmer_recovers_the_one_scatterer_the_measurement_holds() {
        let (baselines, elevations, params) = tomography();
        let truth = -6.0;
        let y = steering(&baselines, params.wavelength, params.slant_range, truth);

        let spectrum = collect(0.0_f64, |out, cap, len| unsafe {
            sl_sl1mmer_spectrum(
                y.as_ptr(),
                baselines.as_ptr(),
                baselines.len(),
                elevations.as_ptr(),
                elevations.len(),
                &raw const params,
                out,
                cap,
                len,
            )
        })
        .expect("the geometry is well posed");
        let peak = elevations[argmax(&spectrum)];
        assert!(
            (peak - truth).abs() < 1e-9,
            "the sparse spectrum peaks at {peak} m, not the {truth} m it was given"
        );

        let scatterers = collect(
            SlScatterer {
                elevation_m: 0.0,
                reflectivity: SlComplex { re: 0.0, im: 0.0 },
            },
            |out, cap, len| unsafe {
                sl_sl1mmer(
                    y.as_ptr(),
                    baselines.as_ptr(),
                    baselines.len(),
                    elevations.as_ptr(),
                    elevations.len(),
                    &raw const params,
                    out,
                    cap,
                    len,
                )
            },
        )
        .expect("the geometry is well posed");
        assert_eq!(
            scatterers.len(),
            1,
            "one scatterer went in and {} came out",
            scatterers.len()
        );
        assert!((scatterers[0].elevation_m - truth).abs() < 1e-9);
        // The l1 step biases the magnitude low and the least-squares refit is what puts it back,
        // so a debiased reflectivity near unity is the property that the refit actually ran.
        let magnitude = scatterers[0]
            .reflectivity
            .re
            .hypot(scatterers[0].reflectivity.im);
        assert!(
            (magnitude - 1.0).abs() < 1e-6,
            "the debiased reflectivity is {magnitude}, not the unit amplitude that went in"
        );

        // Neither count may be zero. This is the vector path, which sizes its data by
        // `baseline_count` directly rather than by its square, so it is the one that has to make
        // the refusal itself.
        for (baseline_count, elevation_count) in [(0, elevations.len()), (baselines.len(), 0)] {
            assert_eq!(
                collect(0.0_f64, |out, cap, len| unsafe {
                    sl_sl1mmer_spectrum(
                        y.as_ptr(),
                        baselines.as_ptr(),
                        baseline_count,
                        elevations.as_ptr(),
                        elevation_count,
                        &raw const params,
                        out,
                        cap,
                        len,
                    )
                })
                .unwrap_err(),
                SL_ERR_INVALID_ARGUMENT,
                "{}",
                last_error()
            );
        }
    }

    #[test]
    fn the_dispersion_is_a_coefficient_of_variation_and_the_selection_is_its_threshold() {
        // Pixel 0 never moves, pixel 1 swings by 80%: the first is a persistent scatterer and the
        // second is not, at any threshold between them.
        let stack = [10.0, 10.0, 10.0, 2.0, 10.0, 18.0];
        let dispersion = collect(0.0_f64, |out, cap, len| unsafe {
            sl_amplitude_dispersion(stack.as_ptr(), 3, 2, out, cap, len)
        })
        .expect("the stack is well posed");
        assert_eq!(dispersion.len(), 2);
        assert!(
            dispersion[0] < 1e-15,
            "a constant pixel disperses by {}",
            dispersion[0]
        );
        // sigma/mu with sigma^2 = (64 + 0 + 64)/3 and mu = 10.
        let expected = (128.0_f64 / 3.0).sqrt() / 10.0;
        assert!(
            (dispersion[1] - expected).abs() < 1e-12,
            "the swinging pixel disperses by {}, not {expected}",
            dispersion[1]
        );

        let selected = collect(0_usize, |out, cap, len| unsafe {
            sl_select_persistent_scatterers(
                dispersion.as_ptr(),
                dispersion.len(),
                0.5,
                out,
                cap,
                len,
            )
        })
        .expect("the dispersion is well posed");
        assert_eq!(selected, vec![0]);
    }

    #[test]
    fn the_sbas_inversion_recovers_the_motion_the_stack_was_built_from() {
        let (wavelength, slant_range, incidence) = (0.056, 800_000.0, 0.6);
        let (velocity, height) = (-0.021, 14.0);
        let temporal = [0.0, 0.6, 1.1, 1.9, 2.4, 3.2];
        let perpendicular = [0.0, 120.0, -80.0, 210.0, -150.0, 60.0];
        let kv = 4.0 * core::f64::consts::PI / wavelength;
        let kh = 4.0 * core::f64::consts::PI / (wavelength * slant_range * incidence.sin());
        let phases: Vec<f64> = temporal
            .iter()
            .zip(&perpendicular)
            .map(|(&dt, &b)| kv * dt * velocity + kh * b * height)
            .collect();

        let mut out = SlDeformation {
            velocity_ms: 0.0,
            height_m: 0.0,
        };
        let status = unsafe {
            sl_sbas_invert(
                phases.as_ptr(),
                temporal.as_ptr(),
                perpendicular.as_ptr(),
                phases.len(),
                wavelength,
                slant_range,
                incidence,
                &raw mut out,
            )
        };
        assert_eq!(status, SL_OK);
        assert!(
            (out.velocity_ms - velocity).abs() < 1e-12,
            "the inversion answered {} m/s, not the {velocity} it was built from",
            out.velocity_ms
        );
        assert!(
            (out.height_m - height).abs() < 1e-9,
            "the inversion answered {} m of topography, not {height}",
            out.height_m
        );
    }

    #[test]
    fn the_coherency_matrix_is_hermitian_and_carries_the_power_that_went_into_it() {
        let samples = [
            SlPolarimetricSample {
                s_hh: SlComplex { re: 1.0, im: 0.2 },
                s_hv: SlComplex { re: 0.1, im: -0.05 },
                s_vv: SlComplex { re: 0.8, im: -0.3 },
            },
            SlPolarimetricSample {
                s_hh: SlComplex { re: 0.9, im: -0.1 },
                s_hv: SlComplex { re: 0.05, im: 0.2 },
                s_vv: SlComplex { re: 1.1, im: 0.15 },
            },
        ];

        let t3 = collect(SlComplex { re: 0.0, im: 0.0 }, |out, cap, len| unsafe {
            sl_coherency_matrix(samples.as_ptr(), samples.len(), out, cap, len)
        })
        .expect("the looks are well posed");
        assert_eq!(t3.len(), 9);
        for i in 0..3 {
            assert!(
                t3[i * 3 + i].im.abs() < 1e-15 && t3[i * 3 + i].re >= 0.0,
                "the diagonal of a coherency matrix is a real power, and entry {i} is not"
            );
            for j in 0..3 {
                let (a, b) = (t3[i * 3 + j], t3[j * 3 + i]);
                assert!(
                    (a.re - b.re).abs() < 1e-15 && (a.im + b.im).abs() < 1e-15,
                    "the matrix is not Hermitian at ({i}, {j})"
                );
            }
        }

        // The trace is the total scattered power, and the Pauli channel powers are the same total
        // split three ways, so the two must agree look by look.
        let trace = t3[0].re + t3[4].re + t3[8].re;
        let mut mean_power = 0.0;
        for sample in &samples {
            let mut powers = [0.0_f64; 3];
            let status = unsafe { sl_pauli_powers(&raw const *sample, powers.as_mut_ptr()) };
            assert_eq!(status, SL_OK);
            mean_power += powers.iter().sum::<f64>() / 2.0;
        }
        assert!(
            (trace - mean_power).abs() < 1e-12,
            "the trace is {trace} and the Pauli powers sum to {mean_power}"
        );
    }

    #[test]
    fn a_scatterer_with_no_preferred_mechanism_is_the_one_with_maximum_entropy() {
        // T3 = I/3 is the completely depolarised case: every eigenvalue equal, so the entropy is
        // its maximum of 1 and the anisotropy -- which measures the gap between the second and
        // third -- is zero. Any implementation that sorts the eigenvalues wrongly fails this.
        let third = 1.0 / 3.0;
        let mut t3 = [SlComplex { re: 0.0, im: 0.0 }; 9];
        for i in 0..3 {
            t3[i * 3 + i] = SlComplex { re: third, im: 0.0 };
        }
        let mut out = SlCloudePottier {
            entropy: 0.0,
            anisotropy: 0.0,
            alpha_rad: 0.0,
        };
        let status = unsafe { sl_cloude_pottier_decompose(t3.as_ptr(), &raw mut out) };
        assert_eq!(status, SL_OK);
        assert!(
            (out.entropy - 1.0).abs() < 1e-12,
            "a depolarised target has entropy {}, not 1",
            out.entropy
        );
        assert!(
            out.anisotropy.abs() < 1e-12,
            "equal eigenvalues cannot be anisotropic, and these are by {}",
            out.anisotropy
        );
    }

    #[test]
    fn a_sinc_response_measures_the_width_and_the_sidelobe_the_textbook_gives_it() {
        // The compressed response of a rectangular spectrum is a sinc, whose 3 dB width is
        // 0.8859 of the Rayleigh spacing and whose first sidelobe sits 13.26 dB down. Both are
        // closed forms, so this pins the metrics against arithmetic rather than against a
        // recorded number.
        let oversampling = 8.0;
        let magnitude: Vec<f64> = (0..512)
            .map(|k| {
                let x = (f64::from(k) - 256.0) / oversampling;
                if x.abs() < 1e-12 {
                    1.0
                } else {
                    let px = core::f64::consts::PI * x;
                    (px.sin() / px).abs()
                }
            })
            .collect();

        let mut width = 0.0;
        assert_eq!(
            unsafe {
                sl_resolution_3db_samples(magnitude.as_ptr(), magnitude.len(), &raw mut width)
            },
            SL_OK
        );
        let expected = 0.885_893_78 * oversampling;
        assert!(
            (width - expected).abs() < 0.02 * expected,
            "the 3 dB width is {width} samples, not the {expected} a sinc has"
        );

        let mut pslr = 0.0;
        assert_eq!(
            unsafe {
                sl_peak_sidelobe_ratio_db(magnitude.as_ptr(), magnitude.len(), &raw mut pslr)
            },
            SL_OK
        );
        assert!(
            (pslr + 13.26).abs() < 0.15,
            "the peak sidelobe is {pslr} dB, not the -13.26 dB a sinc has"
        );
    }

    #[test]
    fn a_profile_with_no_main_lobe_is_refused_rather_than_measured() {
        // A flat profile has a peak that never falls to half power, so there is no width to
        // report. Answering NaN across a C boundary would look like an answer; a status does not.
        let flat = [1.0_f64; 16];
        let mut width = 0.0;
        assert_eq!(
            unsafe { sl_resolution_3db_samples(flat.as_ptr(), flat.len(), &raw mut width) },
            SL_ERR_DOMAIN
        );
        let mut pslr = 0.0;
        assert_eq!(
            unsafe { sl_peak_sidelobe_ratio_db(core::ptr::null(), 0, &raw mut pslr) },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn the_range_doppler_rates_are_the_closed_forms_they_promise() {
        let geometry = SlRdaGeometry {
            prf: 1_700.0,
            velocity: 7_100.0,
            wavelength: 0.055,
            r0_slant: 850_000.0,
            range_spacing: 4.5,
        };
        let rate = unsafe { sl_azimuth_fm_rate(&raw const geometry) };
        let expected =
            2.0 * geometry.velocity * geometry.velocity / (geometry.wavelength * geometry.r0_slant);
        assert!(
            (rate - expected).abs() < 1e-9 * expected,
            "the azimuth FM rate is {rate}, not the 2V^2/(lambda r0) of {expected}"
        );

        // The migration is quadratic in the azimuth frequency, so doubling the frequency has to
        // quadruple it, and at zero Doppler there is no migration at all.
        assert!(unsafe { sl_range_migration(0.0, &raw const geometry) }.abs() < 1e-15);
        let one = unsafe { sl_range_migration(300.0, &raw const geometry) };
        let two = unsafe { sl_range_migration(600.0, &raw const geometry) };
        assert!(
            (two / one - 4.0).abs() < 1e-9,
            "doubling the Doppler multiplied the migration by {}",
            two / one
        );

        // A null geometry is the one case these two total functions cannot answer, and NaN is
        // what a function with no status to return has to say.
        assert!(unsafe { sl_azimuth_fm_rate(core::ptr::null()) }.is_nan());
        assert!(unsafe { sl_range_migration(1.0, core::ptr::null()) }.is_nan());

        // Zero Doppler has no secondary range compression rate -- the coupling it corrects is
        // proportional to f_eta^2, so at the band centre there is nothing to correct.
        let mut src = 0.0;
        assert_eq!(
            unsafe { sl_src_fm_rate(0.0, &raw const geometry, &raw mut src) },
            SL_ERR_DOMAIN
        );
        assert_eq!(
            unsafe { sl_src_fm_rate(400.0, &raw const geometry, &raw mut src) },
            SL_OK
        );
        assert!(src.is_finite() && src != 0.0);
    }

    #[test]
    fn chirp_scaling_is_a_phase_multiply_that_leaves_every_modulus_where_it_was() {
        let line: Vec<SlComplex> = (0..32)
            .map(|k| {
                let phase = f64::from(k) * 0.11;
                SlComplex {
                    re: 1.5 * phase.cos(),
                    im: 1.5 * phase.sin(),
                }
            })
            .collect();
        let scaled = collect(SlComplex { re: 0.0, im: 0.0 }, |out, cap, len| unsafe {
            sl_apply_chirp_scaling(
                line.as_ptr(),
                line.len(),
                1.0e12,
                0.03,
                1.0e-5,
                3.0e7,
                out,
                cap,
                len,
            )
        })
        .expect("the line is well posed");
        assert_eq!(scaled.len(), line.len());
        for (got, want) in scaled.iter().zip(&line) {
            assert!(
                (got.re.hypot(got.im) - want.re.hypot(want.im)).abs() < 1e-12,
                "a phase multiply changed an amplitude"
            );
        }
    }

    #[test]
    fn the_focus_answers_the_aperture_it_was_given_and_refuses_one_it_was_not() {
        let (na, nr) = (8_usize, 16_usize);
        let params = SlCsaParams {
            prf: 1_700.0,
            velocity: 7_100.0,
            wavelength: 0.055,
            r_ref: 850_000.0,
            range_sample_rate: 3.0e7,
            range_fm_rate: 5.0e11,
        };
        let raw: Vec<SlComplex> = (0..na * nr)
            .map(|k| {
                #[allow(clippy::cast_precision_loss)] // 128 samples
                let phase = k as f64 * 0.017;
                SlComplex {
                    re: phase.cos(),
                    im: phase.sin(),
                }
            })
            .collect();
        let focused = collect(SlComplex { re: 0.0, im: 0.0 }, |out, cap, len| unsafe {
            sl_chirp_scaling_focus(raw.as_ptr(), na, nr, &raw const params, out, cap, len)
        })
        .expect("the aperture is well posed");
        assert_eq!(focused.len(), na * nr);

        // The focus is an FFT in both directions, so an aperture whose dimensions are not powers
        // of two is not a smaller problem -- it is a different one, and refusing it is the only
        // answer that does not silently truncate the scene.
        assert_eq!(
            collect(SlComplex { re: 0.0, im: 0.0 }, |out, cap, len| unsafe {
                sl_chirp_scaling_focus(raw.as_ptr(), 6, nr, &raw const params, out, cap, len)
            })
            .unwrap_err(),
            SL_ERR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn a_short_buffer_is_refused_with_the_length_it_needed() {
        let mut len = 0_usize;
        let mut too_small = [SlComplex { re: 0.0, im: 0.0 }; 3];
        let status = unsafe {
            sl_lfm_chirp(
                32,
                5.0e11,
                3.0e7,
                too_small.as_mut_ptr(),
                too_small.len(),
                &raw mut len,
            )
        };
        assert_eq!(status, SL_ERR_BUFFER_TOO_SMALL);
        assert_eq!(len, 32, "the refusal did not name the length it needed");
        assert!(
            too_small.iter().all(|c| c.re == 0.0 && c.im == 0.0),
            "a refused call wrote into the caller's buffer anyway"
        );
    }

    /// A real Pleiades RPC, the same one the earthvision crate verifies its sensor model against.
    const RPB: &str =
        include_str!("../../spacelab-earthvision/tests/fixtures/pleiades_reunion.rpb");

    /// A camera looking down `+z` from `center`, with no rotation -- the simplest rig that still
    /// has a baseline, and the one every geometry test below is built on.
    fn looking_forward(center: [f64; 3], focal: f64, principal: [f64; 2]) -> SlCamera {
        SlCamera {
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            center,
            focal,
            principal,
        }
    }

    /// Where that camera puts a world point -- the collinearity equations, written out here so the
    /// test states the projection it is inverting rather than borrowing it.
    fn project(camera: SlCamera, world: [f64; 3]) -> [f64; 2] {
        let d = [
            world[0] - camera.center[0],
            world[1] - camera.center[1],
            world[2] - camera.center[2],
        ];
        [
            camera.focal * d[0] / d[2] + camera.principal[0],
            camera.focal * d[1] / d[2] + camera.principal[1],
        ]
    }

    #[test]
    fn the_stereo_match_recovers_the_shift_the_pair_was_built_with() {
        // The right image is the left one slid sideways by a known disparity, which is the one
        // case a matcher has no excuse for: the census descriptors are identical at the correct
        // shift, so the cost there is exactly zero.
        let (rows, cols, truth) = (24_usize, 32_usize, 4_usize);
        let texture = |y: usize, x: usize| {
            #[allow(clippy::cast_precision_loss)] // a 24x32 raster
            let (fy, fx) = (y as f64, x as f64);
            (0.9 * fx).sin() + (0.7 * fy).cos() + 0.3 * (0.4 * fx + 0.6 * fy).sin()
        };
        let mut left = vec![0.0; rows * cols];
        let mut right = vec![0.0; rows * cols];
        for y in 0..rows {
            for x in 0..cols {
                left[y * cols + x] = texture(y, x);
                right[y * cols + x] = texture(y, x + truth);
            }
        }

        let config = SlSgmConfig {
            d_max: 8,
            radius: 2,
            p1: 1.0,
            p2: 8.0,
        };
        let disparity = collect(0.0_f64, |out, cap, len| unsafe {
            sl_semi_global_match(
                left.as_ptr(),
                right.as_ptr(),
                rows,
                cols,
                &raw const config,
                out,
                cap,
                len,
            )
        })
        .expect("the pair is well posed");
        assert_eq!(disparity.len(), rows * cols);
        // Only the interior is fair game: a pixel within `truth` of the left edge has no match in
        // the right image at all, and the border census is clamped rather than computed.
        for y in 4..rows - 4 {
            for x in (truth + 4)..cols - 4 {
                let got = disparity[y * cols + x];
                #[allow(clippy::cast_precision_loss)] // a disparity of 4
                let want = truth as f64;
                assert!(
                    (got - want).abs() < 0.6,
                    "at ({y}, {x}) the match says {got}, not the {want} it was built with"
                );
            }
        }

        // The widest window the descriptor does hold is accepted: a 7x7 census is 48 neighbours
        // against 64 bits, so the refusal below is a cliff at 4 and not a slope from 2.
        let widest = SlSgmConfig {
            radius: 3,
            ..config
        };
        assert!(
            collect(0.0_f64, |out, cap, len| unsafe {
                sl_semi_global_match(
                    left.as_ptr(),
                    right.as_ptr(),
                    rows,
                    cols,
                    &raw const widest,
                    out,
                    cap,
                    len,
                )
            })
            .is_ok(),
            "a census radius of 3 fits the descriptor: {}",
            last_error()
        );

        // A census window wider than the descriptor is refused rather than silently truncated.
        let too_wide = SlSgmConfig {
            radius: 4,
            ..config
        };
        assert_eq!(
            collect(0.0_f64, |out, cap, len| unsafe {
                sl_semi_global_match(
                    left.as_ptr(),
                    right.as_ptr(),
                    rows,
                    cols,
                    &raw const too_wide,
                    out,
                    cap,
                    len,
                )
            })
            .unwrap_err(),
            SL_ERR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn the_sweep_finds_the_plane_the_scene_was_rendered_on() {
        // Both images are renderings of one textured plane at a known depth, so the sweep has a
        // correct answer to find and it is on the grid it was given. The baseline is chosen so the
        // depth grid separates: at f = 100 and Z = 10 the parallax is 2 pixels per metre, which is
        // several texture periods across the grid rather than a fraction of one.
        let (rows, cols) = (64_usize, 64_usize);
        let (focal, truth, baseline) = (100.0, 10.0, 2.0);
        let principal = [32.0, 32.0];
        let reference_camera = looking_forward([0.0, 0.0, 0.0], focal, principal);
        let neighbor_camera = looking_forward([baseline, 0.0, 0.0], focal, principal);
        let texture =
            |x: f64, y: f64| (3.0 * x).sin() + (2.3 * y).cos() + 0.5 * (1.7 * x + 2.1 * y).sin();
        let render = |camera: SlCamera| {
            let mut image = vec![0.0; rows * cols];
            for v in 0..rows {
                for u in 0..cols {
                    #[allow(clippy::cast_precision_loss)] // a 64x64 raster
                    let (fu, fv) = (u as f64, v as f64);
                    let x = (fu - camera.principal[0]) * truth / focal + camera.center[0];
                    let y = (fv - camera.principal[1]) * truth / focal + camera.center[1];
                    image[v * cols + u] = texture(x, y);
                }
            }
            image
        };
        let reference_image = render(reference_camera);
        let neighbor_image = render(neighbor_camera);

        let reference = SlMvsView {
            image: reference_image.as_ptr(),
            camera: reference_camera,
        };
        let neighbors = [SlMvsView {
            image: neighbor_image.as_ptr(),
            camera: neighbor_camera,
        }];
        let depths = [7.0, 8.5, 10.0, 11.5, 13.0];
        let depth_map = collect(0.0_f64, |out, cap, len| unsafe {
            sl_plane_sweep_mvs(
                &raw const reference,
                neighbors.as_ptr(),
                neighbors.len(),
                rows,
                cols,
                depths.as_ptr(),
                depths.len(),
                2,
                out,
                cap,
                len,
            )
        })
        .expect("the scene is well posed");
        assert_eq!(depth_map.len(), rows * cols);
        // The neighbour is displaced by 20 pixels at the true depth, so the left fifth of the
        // reference image is simply not in the other picture. Only the overlap is fair game.
        let interior: Vec<f64> = (8..rows - 8)
            .flat_map(|v| (28..cols - 6).map(move |u| (v, u)))
            .map(|(v, u)| depth_map[v * cols + u])
            .collect();
        let found = interior
            .iter()
            .filter(|&&d| (d - truth).abs() < 1e-9)
            .count();
        assert!(
            found * 10 >= interior.len() * 9,
            "the sweep put only {found} of {} overlapping pixels on the plane",
            interior.len()
        );
    }

    #[test]
    fn triangulation_inverts_the_projection_and_refuses_a_rig_with_no_baseline() {
        let focal = 200.0;
        let principal = [50.0, 40.0];
        let cameras = [
            looking_forward([0.0, 0.0, 0.0], focal, principal),
            looking_forward([3.0, 0.0, 0.0], focal, principal),
            looking_forward([0.0, 2.5, 0.0], focal, principal),
        ];
        let truth = [1.2, -0.7, 14.0];
        let pixels: Vec<f64> = cameras.iter().flat_map(|&c| project(c, truth)).collect();

        let mut point = SlVec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        assert_eq!(
            unsafe { sl_triangulate(cameras.as_ptr(), pixels.as_ptr(), 3, &raw mut point) },
            SL_OK
        );
        for (got, want) in [point.x, point.y, point.z].iter().zip(&truth) {
            assert!(
                (got - want).abs() < 1e-9,
                "the intersection is {got}, not the {want} that was projected"
            );
        }

        // Two copies of the same camera see the same ray twice, which intersects nowhere.
        let twice = [cameras[0], cameras[0]];
        let same = [pixels[0], pixels[1], pixels[0], pixels[1]];
        assert_eq!(
            unsafe { sl_triangulate(twice.as_ptr(), same.as_ptr(), 2, &raw mut point) },
            SL_ERR_DOMAIN
        );
        assert_eq!(
            unsafe { sl_triangulate(cameras.as_ptr(), pixels.as_ptr(), 1, &raw mut point) },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn a_track_is_the_matches_it_was_given_and_a_match_to_nowhere_is_refused() {
        let focal = 200.0;
        let principal = [50.0, 40.0];
        let cameras = [
            looking_forward([0.0, 0.0, 0.0], focal, principal),
            looking_forward([3.0, 0.0, 0.0], focal, principal),
        ];
        let truth = [0.5, 0.3, 12.0];
        let corners: Vec<SlCorner> = cameras
            .iter()
            .map(|&c| {
                let p = project(c, truth);
                SlCorner {
                    x: p[0],
                    y: p[1],
                    response: 1.0,
                }
            })
            .collect();
        let counts = [1_usize, 1_usize];
        let matches = [SlFeatureMatch {
            view_a: 0,
            corner_a: 0,
            view_b: 1,
            corner_b: 0,
        }];

        let tracks = collect(
            SlTrackPoint {
                position: SlVec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                resolved: false,
            },
            |out, cap, len| unsafe {
                sl_sfm_triangulate(
                    corners.as_ptr(),
                    counts.as_ptr(),
                    2,
                    matches.as_ptr(),
                    1,
                    cameras.as_ptr(),
                    out,
                    cap,
                    len,
                )
            },
        )
        .expect("the views are well posed");
        assert_eq!(tracks.len(), 1);
        assert!(tracks[0].resolved);
        assert!((tracks[0].position.z - truth[2]).abs() < 1e-9);

        // The track builder indexes the corner lists directly, so a match past their ends is a
        // panic waiting to happen. The boundary is where it is caught, and a status is what a C
        // caller can act on.
        let nowhere = [SlFeatureMatch {
            view_a: 0,
            corner_a: 7,
            view_b: 1,
            corner_b: 0,
        }];
        assert_eq!(
            collect(
                SlTrackPoint {
                    position: SlVec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0
                    },
                    resolved: false,
                },
                |out, cap, len| unsafe {
                    sl_sfm_triangulate(
                        corners.as_ptr(),
                        counts.as_ptr(),
                        2,
                        nowhere.as_ptr(),
                        1,
                        cameras.as_ptr(),
                        out,
                        cap,
                        len,
                    )
                },
            )
            .unwrap_err(),
            SL_ERR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn the_detector_finds_the_corner_that_was_drawn_and_not_the_edge() {
        // A bright quadrant on a dark field has exactly one corner, where the two edges meet. An
        // edge alone must not answer: that is the whole point of the Harris response over the
        // gradient magnitude.
        let (rows, cols) = (24_usize, 24_usize);
        let mut image = vec![0.0; rows * cols];
        for y in 12..rows {
            for x in 12..cols {
                image[y * cols + x] = 1.0;
            }
        }
        let config = SlHarrisConfig {
            k: 0.04,
            window_radius: 2,
            threshold_rel: 0.3,
            nms_radius: 3,
            max_corners: 8,
        };
        let corners = collect(
            SlCorner {
                x: 0.0,
                y: 0.0,
                response: 0.0,
            },
            |out, cap, len| unsafe {
                sl_harris_corners(image.as_ptr(), rows, cols, &raw const config, out, cap, len)
            },
        )
        .expect("the image is well posed");
        assert!(!corners.is_empty(), "a drawn corner went undetected");
        let best = corners
            .iter()
            .max_by(|a, b| a.response.total_cmp(&b.response))
            .expect("non-empty");
        assert!(
            (best.x - 12.0).abs() <= 2.0 && (best.y - 12.0).abs() <= 2.0,
            "the strongest corner is at ({}, {}), not the (12, 12) that was drawn",
            best.x,
            best.y
        );

        // A pure horizontal step is an edge, not a corner, and its response is far weaker.
        let mut edge = vec![0.0; rows * cols];
        for y in 12..rows {
            for x in 0..cols {
                edge[y * cols + x] = 1.0;
            }
        }
        let on_edge = collect(
            SlCorner {
                x: 0.0,
                y: 0.0,
                response: 0.0,
            },
            |out, cap, len| unsafe {
                sl_harris_corners(edge.as_ptr(), rows, cols, &raw const config, out, cap, len)
            },
        )
        .expect("the image is well posed");
        let strongest = on_edge.iter().map(|c| c.response).fold(0.0_f64, f64::max);
        assert!(
            strongest < best.response,
            "an edge answered {strongest}, at least as strongly as the corner's {}",
            best.response
        );
    }

    #[test]
    fn the_block_adjustment_absorbs_the_bias_the_control_points_carry() {
        // Two exact properties of a shift fit, neither of which needs a second implementation of
        // the rational polynomial: one control point is absorbed exactly, and moving every
        // measured pixel by a constant moves the corrected forward by the same constant.
        let ground = [
            (-21.10, 55.40, 100.0),
            (-21.14, 55.44, 250.0),
            (-21.06, 55.36, 400.0),
            (-21.12, 55.48, 50.0),
        ];
        let query = (-21.09, 55.42, 180.0);

        let adjust = |control: &[SlGroundControlPoint], at: (f64, f64, f64), affine: bool| {
            let mut out = SlImagePoint { row: 0.0, col: 0.0 };
            let status = unsafe {
                sl_rpc_block_adjust(
                    RPB.as_ptr().cast(),
                    RPB.len(),
                    control.as_ptr(),
                    control.len(),
                    at.0,
                    at.1,
                    at.2,
                    affine,
                    &raw mut out,
                )
            };
            (status, out)
        };

        // One point, one shift: the correction has exactly the residual to absorb, so the
        // corrected forward lands on the measurement itself.
        let single = [SlGroundControlPoint {
            latitude: ground[0].0,
            longitude: ground[0].1,
            height: ground[0].2,
            row: 1234.5,
            col: 6789.25,
        }];
        let (status, at) = adjust(&single, ground[0], false);
        assert_eq!(status, SL_OK);
        assert!(
            (at.row - single[0].row).abs() < 1e-9 && (at.col - single[0].col).abs() < 1e-9,
            "a single control point was not absorbed: ({}, {})",
            at.row,
            at.col
        );

        // Four points, and the same four moved bodily: the fit is affine in the measurements, so
        // the answer moves by exactly the offset that was put in.
        let (drow, dcol) = (4.5, -2.25);
        let control: Vec<SlGroundControlPoint> = ground
            .iter()
            .enumerate()
            .map(|(i, &(latitude, longitude, height))| SlGroundControlPoint {
                latitude,
                longitude,
                height,
                #[allow(clippy::cast_precision_loss)] // four control points
                row: 1000.0 + i as f64,
                #[allow(clippy::cast_precision_loss)] // four control points
                col: 2000.0 - i as f64,
            })
            .collect();
        let moved: Vec<SlGroundControlPoint> = control
            .iter()
            .map(|p| SlGroundControlPoint {
                row: p.row + drow,
                col: p.col + dcol,
                ..*p
            })
            .collect();
        let (_, before) = adjust(&control, query, false);
        let (_, after) = adjust(&moved, query, false);
        assert!(
            (after.row - before.row - drow).abs() < 1e-9
                && (after.col - before.col - dcol).abs() < 1e-9,
            "the adjustment moved by ({}, {}), not the ({drow}, {dcol}) that was put in",
            after.row - before.row,
            after.col - before.col
        );

        // The crate asserts on an empty control set and the affine model has six unknowns, so
        // both are refused here rather than reaching a panic the C caller cannot catch.
        assert_eq!(adjust(&[], query, false).0, SL_ERR_INVALID_ARGUMENT);
        assert_eq!(
            adjust(&control[..2], query, true).0,
            SL_ERR_INVALID_ARGUMENT
        );

        let mut out = SlImagePoint { row: 0.0, col: 0.0 };
        assert_eq!(
            unsafe {
                sl_rpc_block_adjust(
                    c"not an RPB".as_ptr(),
                    10,
                    control.as_ptr(),
                    control.len(),
                    query.0,
                    query.1,
                    query.2,
                    false,
                    &raw mut out,
                )
            },
            SL_ERR_PARSE
        );
    }

    #[test]
    fn the_reflectance_is_its_kernels_weighted_and_the_two_atmospheres_are_inverses() {
        for &(theta_s, theta_v, phi) in &[
            (0.3_f64, 0.5_f64, 0.7_f64),
            (0.9, 0.2, 2.4),
            (0.0, 0.0, 0.0),
            (1.2, 1.1, core::f64::consts::PI),
        ] {
            let ross = sl_ross_thick_kernel(theta_s, theta_v, phi);
            let li = sl_li_sparse_reciprocal_kernel(theta_s, theta_v, phi);
            let (f_iso, f_vol, f_geo) = (0.12, 0.03, 0.008);
            let combined = sl_brdf_reflectance(f_iso, f_vol, f_geo, theta_s, theta_v, phi);
            assert!(
                (combined - (f_iso + f_vol * ross + f_geo * li)).abs() < 1e-15,
                "the RTLSR sum is not its own kernels at ({theta_s}, {theta_v}, {phi})"
            );
        }

        let coupling = SlAtmosphericCoupling {
            path_reflectance: 0.08,
            transmittance_down: 0.83,
            transmittance_up: 0.87,
            spherical_albedo: 0.11,
        };
        for surface in [0.0_f64, 0.05, 0.2, 0.6, 0.95] {
            let toa = unsafe { sl_atmospheric_toa_from_surface(&raw const coupling, surface) };
            let back = unsafe { sl_atmospheric_surface_from_toa(&raw const coupling, toa) };
            assert!(
                (back - surface).abs() < 1e-12,
                "the 6S round trip turned {surface} into {back}"
            );
        }
        assert!(unsafe { sl_atmospheric_toa_from_surface(core::ptr::null(), 0.3) }.is_nan());
        assert!(unsafe { sl_atmospheric_surface_from_toa(core::ptr::null(), 0.3) }.is_nan());
    }

    #[test]
    fn brovey_puts_the_panchromatic_total_back_and_gram_schmidt_keeps_the_shape() {
        let pixels = 6;
        let bands = [
            0.20, 0.31, 0.15, 0.44, 0.27, 0.0, //
            0.35, 0.22, 0.51, 0.18, 0.33, 0.0, //
            0.11, 0.47, 0.26, 0.29, 0.40, 0.0,
        ];
        let pan = [0.55, 0.61, 0.48, 0.72, 0.66, 0.39];

        let brovey = collect(0.0_f64, |out, cap, len| unsafe {
            sl_brovey_pan_sharpen(bands.as_ptr(), 3, pixels, pan.as_ptr(), out, cap, len)
        })
        .expect("the bands are well posed");
        assert_eq!(brovey.len(), 3 * pixels);
        for p in 0..pixels {
            let total: f64 = (0..3).map(|k| brovey[k * pixels + p]).sum();
            // Brovey is a ratio substitution, so the sharpened bands sum back to the pan value --
            // except where the multispectral bands sum to zero and there is no ratio to take.
            let want = if p == 5 { 0.0 } else { pan[p] };
            assert!(
                (total - want).abs() < 1e-12,
                "pixel {p} sums to {total}, not the {want} it should"
            );
        }

        let gram_schmidt = collect(0.0_f64, |out, cap, len| unsafe {
            sl_gram_schmidt_pan_sharpen(bands.as_ptr(), 3, pixels, pan.as_ptr(), out, cap, len)
        })
        .expect("the bands are well posed");
        assert_eq!(gram_schmidt.len(), 3 * pixels);
        assert!(gram_schmidt.iter().all(|v| v.is_finite()));
        // The two are different substitutions of the same detail, so they must not coincide.
        assert!(
            gram_schmidt
                .iter()
                .zip(&brovey)
                .any(|(g, b)| (g - b).abs() > 1e-6),
            "the two pan-sharpeners answered the same thing"
        );
    }

    #[test]
    fn the_overlap_is_what_suppression_suppresses_on() {
        let unit = SlBbox {
            x1: 0.0,
            y1: 0.0,
            x2: 2.0,
            y2: 2.0,
        };
        let half = SlBbox {
            x1: 1.0,
            y1: 0.0,
            x2: 3.0,
            y2: 2.0,
        };
        let apart = SlBbox {
            x1: 10.0,
            y1: 10.0,
            x2: 12.0,
            y2: 12.0,
        };
        assert!((unsafe { sl_box_iou(&raw const unit, &raw const unit) } - 1.0).abs() < 1e-15);
        assert!(unsafe { sl_box_iou(&raw const unit, &raw const apart) }.abs() < 1e-15);
        // Two unit-area-4 squares overlapping in half: intersection 2, union 6.
        assert!(
            (unsafe { sl_box_iou(&raw const unit, &raw const half) } - 1.0 / 3.0).abs() < 1e-15
        );
        assert!(unsafe { sl_box_iou(core::ptr::null(), &raw const unit) }.is_nan());

        // At a threshold below their overlap the weaker of the pair goes; the distant box stays.
        let boxes = [unit, half, apart];
        let scores = [0.9, 0.8, 0.7];
        let kept = collect(0_usize, |out, cap, len| unsafe {
            sl_nms(boxes.as_ptr(), scores.as_ptr(), 3, 0.3, out, cap, len)
        })
        .expect("the boxes are well posed");
        assert_eq!(kept, vec![0, 2]);
        // Above it, nothing is suppressed at all.
        let all = collect(0_usize, |out, cap, len| unsafe {
            sl_nms(boxes.as_ptr(), scores.as_ptr(), 3, 0.9, out, cap, len)
        })
        .expect("the boxes are well posed");
        assert_eq!(all, vec![0, 1, 2]);
    }

    #[test]
    fn the_anchor_grid_is_the_shape_it_promises_and_a_zero_delta_decodes_to_itself() {
        let scales = [4.0, 8.0];
        let ratios = [0.5, 1.0, 2.0];
        let anchors = collect(
            SlBbox {
                x1: 0.0,
                y1: 0.0,
                x2: 0.0,
                y2: 0.0,
            },
            |out, cap, len| unsafe {
                sl_generate_anchors(
                    3,
                    4,
                    16.0,
                    scales.as_ptr(),
                    scales.len(),
                    ratios.as_ptr(),
                    ratios.len(),
                    out,
                    cap,
                    len,
                )
            },
        )
        .expect("the grid is well posed");
        assert_eq!(anchors.len(), 3 * 4 * scales.len() * ratios.len());
        // Every anchor of a square scale has the area that scale names, whatever its ratio.
        for anchor in &anchors {
            let (w, h) = (anchor.x2 - anchor.x1, anchor.y2 - anchor.y1);
            assert!(w > 0.0 && h > 0.0, "an anchor has no extent: {anchor:?}");
        }

        // The centre parameterisation is affine in the deltas, so a zero delta is the identity.
        let anchor = anchors[5];
        let zero = [0.0_f64; 4];
        let mut decoded = SlBbox {
            x1: 0.0,
            y1: 0.0,
            x2: 0.0,
            y2: 0.0,
        };
        assert_eq!(
            unsafe { sl_decode_box(&raw const anchor, zero.as_ptr(), &raw mut decoded) },
            SL_OK
        );
        for (got, want) in [decoded.x1, decoded.y1, decoded.x2, decoded.y2]
            .iter()
            .zip(&[anchor.x1, anchor.y1, anchor.x2, anchor.y2])
        {
            assert!(
                (got - want).abs() < 1e-12,
                "a zero delta moved the anchor from {want} to {got}"
            );
        }
    }

    /// A real operational VMF3 grid, the same one the GNSS crate interpolates in its own tests.
    const VMF_GRID: &str = include_str!("../../spacelab-gnss/tests/fixtures/vmf3_5x5_20181125.H00");

    #[test]
    fn the_blind_troposphere_and_the_measured_grid_agree_on_the_same_point() {
        // Vienna, roughly: the grid is a 2018 November analysis and GPT3 is a climatology, so they
        // are two different statements about the same place. That they agree to within a factor of
        // two on the mapping coefficients, and that both land in the physical range, is the
        // property -- equality would mean one of them was not doing its job.
        let (latitude, longitude) = (48.2_f64.to_radians(), 16.4_f64.to_radians());

        let mut weather = SlGpt3Weather {
            ah: 0.0,
            aw: 0.0,
            pressure: 0.0,
            temperature: 0.0,
            water_vapor_pressure: 0.0,
            tm: 0.0,
            lambda: 0.0,
            gnh: 0.0,
            geh: 0.0,
            gnw: 0.0,
            gew: 0.0,
        };
        assert_eq!(
            unsafe { sl_gpt3(latitude, longitude, 329.0, 180.0, &raw mut weather) },
            SL_OK
        );
        assert!(
            (80_000.0..110_000.0).contains(&weather.pressure),
            "a surface pressure of {} Pa is not one this planet has",
            weather.pressure
        );
        assert!(
            (220.0..320.0).contains(&weather.temperature) && weather.tm < weather.temperature,
            "the temperatures are {} K surface and {} K weighted mean",
            weather.temperature,
            weather.tm
        );
        assert!(weather.ah > 0.0 && weather.aw > 0.0);

        let mut grid = SlVmfCoefficients {
            ah: 0.0,
            aw: 0.0,
            zhd: 0.0,
            zwd: 0.0,
        };
        assert_eq!(
            unsafe {
                sl_vmf_grid_coefficients(
                    VMF_GRID.as_ptr().cast(),
                    VMF_GRID.len(),
                    latitude,
                    longitude,
                    &raw mut grid,
                )
            },
            SL_OK
        );
        assert!(
            grid.ah > 0.5 * weather.ah && grid.ah < 2.0 * weather.ah,
            "the analysis says a_h = {} where the climatology says {}",
            grid.ah,
            weather.ah
        );
        assert!(
            (1.5..3.0).contains(&grid.zhd) && (0.0..1.0).contains(&grid.zwd),
            "the zenith delays are {} m hydrostatic and {} m wet",
            grid.zhd,
            grid.zwd
        );

        // Text that is not a grid is a parse failure, not an interpolation of nothing.
        assert_eq!(
            unsafe {
                sl_vmf_grid_coefficients(
                    c"not a grid".as_ptr(),
                    10,
                    latitude,
                    longitude,
                    &raw mut grid,
                )
            },
            SL_ERR_PARSE
        );
        assert_eq!(
            unsafe { sl_gpt3(latitude, longitude, 329.0, 180.0, core::ptr::null_mut()) },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    /// The two conditions a cloud-mask call has to meet before a model is loaded or run, and the
    /// number each of them names. Asserted on the helpers rather than through the entry points
    /// because that is where they live, and they live there on purpose: `cargo mutants` generates
    /// nothing for the body of an `unsafe fn`, so a condition written inline in one is outside the
    /// seal (#260).
    #[test]
    fn the_cloud_mask_preconditions_name_the_number_that_is_wrong() {
        const WHO: &str = "who";
        assert_eq!(check_cloud_model(WHO, 1, 0.5), Ok(()));
        // A cut outside [0, 1] is how a caller asks for "nothing is cloudy" or "everything is",
        // so it is an answer rather than a mistake and must not be refused.
        assert_eq!(check_cloud_model(WHO, 1, -1.0), Ok(()));
        assert_eq!(check_cloud_model(WHO, 1, 1.5), Ok(()));
        assert_eq!(check_cloud_model(WHO, 0, 0.5), Err(SL_ERR_INVALID_ARGUMENT));
        assert!(last_error().contains("zero bytes"), "{}", last_error());
        for cut in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(check_cloud_model(WHO, 1, cut), Err(SL_ERR_INVALID_ARGUMENT));
            assert!(last_error().contains("threshold"), "{}", last_error());
        }

        assert_eq!(check_cloud_tile(WHO, 3, 8, 8), Ok(192));
        assert_eq!(check_cloud_tile(WHO, 1, 1, 1), Ok(1));
        for (channels, height, width, named) in [
            (0, 8, 8, "channels"),
            (3, 0, 8, "height"),
            (3, 8, 0, "width"),
        ] {
            assert_eq!(
                check_cloud_tile(WHO, channels, height, width),
                Err(SL_ERR_INVALID_ARGUMENT)
            );
            assert!(last_error().contains(named), "{}", last_error());
        }
        // The product is what indexes the tile, so it has to fit the pointer that indexes it.
        assert_eq!(
            check_cloud_tile(WHO, usize::MAX, 2, 1),
            Err(SL_ERR_INVALID_ARGUMENT)
        );
        assert!(last_error().contains("overflows"), "{}", last_error());
    }

    /// A null is refused before anything is loaded, a refused open leaves the caller's pointer
    /// alone, and closing nothing is not an error -- so a failed open needs no special path.
    #[test]
    fn the_cloud_mask_entry_points_refuse_a_null_and_close_a_null_safely() {
        let model = [0_u8; 4];
        let tile = [0.0_f32; 3];
        let mut handle: *mut SlCloudMasker = core::ptr::null_mut();
        let mut len = 7_usize;
        unsafe {
            assert_eq!(
                sl_cloud_masker_open(core::ptr::null(), 4, 0.5, &raw mut handle),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_cloud_masker_open(model.as_ptr(), 4, 0.5, core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_cloud_masker_mask(
                    core::ptr::null(),
                    tile.as_ptr(),
                    1,
                    1,
                    3,
                    core::ptr::null_mut(),
                    0,
                    &raw mut len,
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            sl_cloud_masker_close(core::ptr::null_mut());
        }
        assert!(handle.is_null(), "a refused open must not write `out`");
        assert_eq!(len, 7, "a refused call must not write `out_len`");
    }

    /// Without the feature the four symbols are still here -- that is the point of exporting them
    /// unconditionally -- and every one of them says which build flag it is missing.
    #[cfg(not(feature = "onnx"))]
    #[test]
    fn without_the_onnx_feature_the_symbols_are_present_and_name_the_flag() {
        let model = [0_u8; 4];
        let tile = [0.0_f32; 3];
        let mut handle: *mut SlCloudMasker = core::ptr::null_mut();
        let mut len = 0_usize;
        assert_eq!(
            unsafe { sl_cloud_masker_open(model.as_ptr(), model.len(), 0.5, &raw mut handle) },
            SL_ERR_UNSUPPORTED
        );
        assert!(last_error().contains("`onnx`"), "{}", last_error());
        assert!(handle.is_null());
        assert_eq!(
            unsafe {
                sl_cloud_mask(
                    model.as_ptr(),
                    model.len(),
                    0.5,
                    tile.as_ptr(),
                    1,
                    1,
                    3,
                    core::ptr::null_mut(),
                    0,
                    &raw mut len,
                )
            },
            SL_ERR_UNSUPPORTED
        );
    }

    /// The mask is the model's own probabilities against the cut, and its length is the model's
    /// output grid rather than the tile's.
    ///
    /// The fixture is a 3x8x8 tile through a net that halves the grid: **16** outputs against 64
    /// input pixels, which is why this ABI cannot let the caller assume `height * width` and why
    /// the two-call protocol is here at all. The expected mask comes from onnxruntime's recorded
    /// output for the same fixture, so it is derived outside this code path.
    #[cfg(feature = "onnx")]
    #[test]
    fn the_cloud_mask_is_the_model_s_probabilities_against_the_cut() {
        fn read_f32(bytes: &[u8]) -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        }
        const ONNX: &[u8] = include_bytes!("../../spacelab-vision/tests/fixtures/cloud_tiny.onnx");
        let tile = read_f32(include_bytes!(
            "../../spacelab-vision/tests/fixtures/cloud_tiny_input.f32"
        ));
        let reference = read_f32(include_bytes!(
            "../../spacelab-vision/tests/fixtures/cloud_tiny_output.f32"
        ));
        assert_eq!(reference.len(), 16);
        let cut = 0.5_f32;
        let want: Vec<u8> = reference.iter().map(|&p| u8::from(p >= cut)).collect();
        assert!(
            want.contains(&0) && want.contains(&1),
            "the fixture must straddle the cut to say anything about a threshold"
        );

        let mut handle: *mut SlCloudMasker = core::ptr::null_mut();
        assert_eq!(
            unsafe { sl_cloud_masker_open(ONNX.as_ptr(), ONNX.len(), cut, &raw mut handle) },
            SL_OK
        );
        assert!(!handle.is_null());

        // Sizing pass: the length is the model's, not the tile's.
        let mut len = 0_usize;
        assert_eq!(
            unsafe {
                sl_cloud_masker_mask(
                    handle,
                    tile.as_ptr(),
                    3,
                    8,
                    8,
                    core::ptr::null_mut(),
                    0,
                    &raw mut len,
                )
            },
            SL_OK
        );
        assert_eq!(len, 16);

        // A buffer shorter than that is refused, and told how short it was.
        let mut short = [0_u8; 4];
        len = 0;
        assert_eq!(
            unsafe {
                sl_cloud_masker_mask(
                    handle,
                    tile.as_ptr(),
                    3,
                    8,
                    8,
                    short.as_mut_ptr(),
                    4,
                    &raw mut len,
                )
            },
            SL_ERR_BUFFER_TOO_SMALL
        );
        assert_eq!(len, 16);

        let mut got = [0_u8; 16];
        len = 0;
        assert_eq!(
            unsafe {
                sl_cloud_masker_mask(
                    handle,
                    tile.as_ptr(),
                    3,
                    8,
                    8,
                    got.as_mut_ptr(),
                    16,
                    &raw mut len,
                )
            },
            SL_OK
        );
        assert_eq!(len, 16);
        assert_eq!(got.as_slice(), want.as_slice());

        // A shape the model cannot accept is a compute failure, not a silent empty mask.
        len = 0;
        assert_eq!(
            unsafe {
                sl_cloud_masker_mask(
                    handle,
                    tile.as_ptr(),
                    3,
                    8,
                    9,
                    got.as_mut_ptr(),
                    16,
                    &raw mut len,
                )
            },
            SL_ERR_COMPUTE
        );

        unsafe { sl_cloud_masker_close(handle) };

        // The one-shot form is the same three calls, and answers the same.
        let mut once = [0_u8; 16];
        len = 0;
        assert_eq!(
            unsafe {
                sl_cloud_mask(
                    ONNX.as_ptr(),
                    ONNX.len(),
                    cut,
                    tile.as_ptr(),
                    3,
                    8,
                    8,
                    once.as_mut_ptr(),
                    16,
                    &raw mut len,
                )
            },
            SL_OK
        );
        assert_eq!(len, 16);
        assert_eq!(once.as_slice(), want.as_slice());

        // Bytes that are not a model are a parse failure, named as one.
        let mut handle2: *mut SlCloudMasker = core::ptr::null_mut();
        assert_eq!(
            unsafe {
                sl_cloud_masker_open(b"not an onnx graph".as_ptr(), 17, cut, &raw mut handle2)
            },
            SL_ERR_PARSE
        );
        assert!(handle2.is_null());
    }

    // ── #319: the entry points #310-#314 and #322 reach the ABI through ──────────────────────

    /// A ground terminal off every axis, so no component of a geometry is zero by accident and a
    /// sign flip has somewhere to show.
    fn terminal_at(x: f64, y: f64, z: f64, epoch: f64) -> SlTerminal {
        SlTerminal {
            position: SlVec3 { x, y, z },
            velocity: SlVec3 {
                x: -7.292_115e-5 * y,
                y: 7.292_115e-5 * x,
                z: 0.0,
            },
            epoch,
        }
    }

    /// Read the one `double` an entry point reports through a status, or fail naming the refusal.
    fn one(call: impl FnOnce(*mut f64) -> i32) -> f64 {
        let mut out = f64::NAN;
        assert_eq!(call(&raw mut out), SL_OK, "{}", last_error());
        out
    }

    /// The bistatic, TDOA/FDOA, ΔDOR and inter-satellite bindings against their crate functions.
    ///
    /// Bit-exact rather than to a tolerance: every wiring mistake available here — the two
    /// terminals swapped, the bounce epoch read as the transmit epoch, the satellite's velocity
    /// dropped — produces a number that is only a little wrong, and a tolerance admits all three.
    #[test]
    fn the_multistatic_bindings_are_their_crate_functions() {
        let sat = leo();
        let six = to_six(sat);
        let tx = terminal_at(4_194_303.0, 172_460.0, 4_780_100.0, 0.0);
        let rx = terminal_at(-3_950_200.0, 2_522_400.0, 4_311_700.0, 0.04);
        let bounce = 0.02;
        let geometry = BistaticGeometry::new(to_terminal(tx), bounce, to_terminal(rx))
            .expect("the epochs increase along the path");
        // SAFETY: every pointer is to a local of the right type.
        unsafe {
            assert_eq!(
                one(|out| sl_bistatic_range(
                    &raw const sat,
                    &raw const tx,
                    bounce,
                    &raw const rx,
                    out
                )),
                estimation_bistatic_range(&six, &geometry)
            );
            assert_eq!(
                one(|out| sl_bistatic_range_rate(
                    &raw const sat,
                    &raw const tx,
                    bounce,
                    &raw const rx,
                    out
                )),
                estimation_bistatic_range_rate(&six, &geometry)
            );
        }

        let emission = 0.0;
        let first = terminal_at(4_194_303.0, 172_460.0, 4_780_100.0, 0.03);
        let second = terminal_at(-3_950_200.0, 2_522_400.0, 4_311_700.0, 0.05);
        let pair = TdoaGeometry::new(emission, to_terminal(first), to_terminal(second))
            .expect("both terminals receive after emission");
        let quasar = Quasar {
            right_ascension: 1.2,
            declination: -0.4,
        };
        // SAFETY: every pointer is to a local of the right type.
        let forward = unsafe {
            let tdoa = one(|out| {
                sl_tdoa(
                    &raw const sat,
                    emission,
                    &raw const first,
                    &raw const second,
                    out,
                )
            });
            assert_eq!(tdoa, estimation_tdoa(&six, &pair));
            assert_eq!(
                one(|out| sl_fdoa(
                    &raw const sat,
                    emission,
                    &raw const first,
                    &raw const second,
                    8.4e9,
                    out
                )),
                estimation_fdoa(&six, &pair, 8.4e9)
            );
            assert_eq!(
                one(|out| sl_delta_dor(
                    &raw const sat,
                    emission,
                    &raw const first,
                    &raw const second,
                    quasar.right_ascension,
                    quasar.declination,
                    out
                )),
                estimation_delta_dor(&six, &pair, quasar)
            );
            tdoa
        };
        // Swapping the receivers negates the difference of arrival. A binding that read `first`
        // twice would answer zero here and would pass every assertion above.
        // SAFETY: every pointer is to a local of the right type.
        let swapped = unsafe {
            one(|out| {
                sl_tdoa(
                    &raw const sat,
                    emission,
                    &raw const second,
                    &raw const first,
                    out,
                )
            })
        };
        assert_eq!(swapped, -forward);

        let remote = SlState {
            position: SlVec3 {
                x: sat.position.x + 12_000.0,
                y: sat.position.y - 3_400.0,
                z: sat.position.z + 900.0,
            },
            velocity: SlVec3 {
                x: sat.velocity.x + 1.5,
                y: sat.velocity.y,
                z: sat.velocity.z - 0.4,
            },
        };
        let other = to_six(remote);
        // SAFETY: every pointer is to a local of the right type, and a null pointer is exactly the
        // input these two are documented to accept.
        unsafe {
            assert_eq!(
                sl_inter_satellite_range(&raw const sat, &raw const remote),
                estimation_inter_satellite_range(&six, &other)
            );
            assert_eq!(
                sl_inter_satellite_range_rate(&raw const sat, &raw const remote),
                estimation_inter_satellite_range_rate(&six, &other)
            );
            assert!(sl_inter_satellite_range(core::ptr::null(), &raw const remote).is_nan());
            assert!(sl_inter_satellite_range_rate(&raw const sat, core::ptr::null()).is_nan());
        }

        // A receiver that hears the bounce before it happened is a malformed request, refused
        // rather than answered with a negative leg.
        let early = terminal_at(-3_950_200.0, 2_522_400.0, 4_311_700.0, bounce - 1.0);
        let mut slot = 0.0;
        // SAFETY: every pointer is to a local of the right type.
        let status = unsafe {
            sl_bistatic_range(
                &raw const sat,
                &raw const tx,
                bounce,
                &raw const early,
                &raw mut slot,
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
        assert!(
            last_error().contains("sl_bistatic_range"),
            "{}",
            last_error()
        );
    }

    /// Moyer's turn-around ratios, and the identity that ties the two Doppler observables together.
    ///
    /// Over a flat exciter referred to its own band, the averaged Doppler and the measured
    /// frequency sum to `M · f_reference` exactly. That is one equation over two entry points, so a
    /// ratio or a sign wrong in either breaks it — which a single value compared against a
    /// separately recomputed constant would not catch, since the constant would be recomputed the
    /// same wrong way.
    #[test]
    fn the_dsn_bindings_carry_moyers_ratios_and_the_count_identity() {
        let exciter = 7.15e9;
        let segments = [SlRampSegment {
            start: 0.0,
            end: 1e4,
            start_frequency: exciter,
            rate: 0.0,
        }];
        let owned = [RampSegment {
            start: 0.0,
            end: 1e4,
            start_frequency: exciter,
            rate: 0.0,
        }];
        let table = RampTable::new(&owned).expect("one forward segment is a table");

        // `C = 26` puts the modulus `2^32` above both bands' answers, so the ratio between them is
        // the ratio of the two conversion factors and nothing else.
        // SAFETY: `segments` is a live local array of exactly `len` elements.
        let (wide, s_band) = unsafe {
            (
                one(|out| {
                    sl_dsn_sequential_range(
                        segments.as_ptr(),
                        segments.len(),
                        BAND_X,
                        100.0,
                        101.0,
                        26,
                        out,
                    )
                }),
                one(|out| {
                    sl_dsn_sequential_range(
                        segments.as_ptr(),
                        segments.len(),
                        BAND_S,
                        100.0,
                        101.0,
                        26,
                        out,
                    )
                }),
            )
        };
        assert_eq!(
            wide,
            estimation_sequential_range(&table, Band::X, 100.0, 101.0, 26)
                .expect("the table covers the interval")
        );
        // Moyer's conversion is `221/2d`, with `d = 749` for X and `221` for S, so the S-band
        // factor is exactly one half and the ratio is `749/221`.
        assert!(
            ((wide - (221.0 / 1498.0) * exciter) / wide).abs() < 1e-12,
            "{wide}"
        );
        assert!(
            ((s_band / wide) - (749.0 / 221.0)).abs() < 1e-12,
            "{s_band} against {wide}"
        );

        let interval = 60.0;
        let link = SlDopplerLink {
            uplink: BAND_X,
            downlink: BAND_X,
            reference_frequency: exciter,
            reference_band: BAND_X,
            count_interval: interval,
            subtracted: true,
        };
        // M(X, X) from Moyer's table 13-1: 880 up, 749 down.
        let turnaround = 880.0 / 749.0;
        for rate in [1.0, 1.0 - 1e-5, 1.0 + 3e-6] {
            // SAFETY: both pointers are to live locals of the right type and extent.
            let (counted, instantaneous) = unsafe {
                (
                    one(|out| {
                        sl_dsn_averaged_doppler(
                            segments.as_ptr(),
                            segments.len(),
                            &raw const link,
                            100.0,
                            interval.mul_add(rate, 100.0),
                            out,
                        )
                    }),
                    one(|out| {
                        sl_dsn_measured_frequency(
                            segments.as_ptr(),
                            segments.len(),
                            BAND_X,
                            BAND_X,
                            100.0,
                            rate,
                            out,
                        )
                    }),
                )
            };
            let sum = counted + instantaneous;
            assert!(
                ((sum - turnaround * exciter) / (turnaround * exciter)).abs() < 1e-13,
                "{sum} is not {} Hz at rate {rate}",
                turnaround * exciter
            );
        }
        // The reference band travels separately from the transmitted one, and it is not decoration:
        // referring the same receiver to S-band scales only the reference term.
        let mut against_s = link;
        against_s.reference_band = BAND_S;
        // SAFETY: both pointers are to live locals of the right type and extent.
        let (s_referred, x_referred) = unsafe {
            (
                one(|out| {
                    sl_dsn_averaged_doppler(
                        segments.as_ptr(),
                        segments.len(),
                        &raw const against_s,
                        100.0,
                        100.0 + interval,
                        out,
                    )
                }),
                one(|out| {
                    sl_dsn_averaged_doppler(
                        segments.as_ptr(),
                        segments.len(),
                        &raw const link,
                        100.0,
                        100.0 + interval,
                        out,
                    )
                }),
            )
        };
        let expected = (880.0 / 221.0 - turnaround) * exciter;
        assert!(
            ((s_referred - x_referred - expected) / expected).abs() < 1e-13,
            "{s_referred} against {x_referred}"
        );

        // A band tag the ABI does not define is refused by name rather than silently taken as S.
        let mut slot = 0.0;
        // SAFETY: `segments` is a live local array; `slot` is a live local.
        let status = unsafe {
            sl_dsn_sequential_range(
                segments.as_ptr(),
                segments.len(),
                7,
                100.0,
                101.0,
                26,
                &raw mut slot,
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
        assert!(last_error().contains("sl_band"), "{}", last_error());

        // A ramp table that does not reach the interval is out of the observable's domain, which is
        // a different refusal from a table that was never well formed.
        // SAFETY: as above.
        let status = unsafe {
            sl_dsn_sequential_range(
                segments.as_ptr(),
                segments.len(),
                BAND_X,
                1e5,
                1e5 + 1.0,
                26,
                &raw mut slot,
            )
        };
        assert_eq!(status, SL_ERR_DOMAIN);
    }

    /// The corona series and the Shapiro delay, each against what it is defined to be.
    #[test]
    fn the_corona_and_shapiro_bindings_are_their_crate_functions() {
        // Ten solar radii off the Sun-probe line, so the ray clears the photosphere by a factor of
        // ten and the closed form is evaluated well inside its domain.
        let impact = 10.0 * SOLAR_RADIUS;
        let tx = SlVec3 {
            x: -1.5e11,
            y: impact,
            z: 0.0,
        };
        let rx = SlVec3 {
            x: 2.3e11,
            y: impact,
            z: 0.0,
        };
        let model =
            CoronaSeries::new(&AKSIM_PAVLOV_2022).expect("the published series is a series");
        // SAFETY: both position pointers are to live locals; `series` is null, which is the
        // documented way to ask for the default.
        let delay = unsafe {
            one(|out| {
                sl_solar_corona_delay(
                    &raw const tx,
                    &raw const rx,
                    8.4e9,
                    true,
                    core::ptr::null(),
                    0,
                    out,
                )
            })
        };
        assert_eq!(
            delay,
            estimation_solar_corona_delay(
                &model,
                &to_three(tx),
                &to_three(rx),
                8.4e9,
                PlasmaSignature::GroupDelay
            )
            .expect("the ray clears the photosphere")
        );
        assert!(delay > 0.0, "a group delay is a delay: {delay}");

        // A phase observable arrives early by exactly the magnitude the group delay arrives late.
        // SAFETY: as above.
        let phase = unsafe {
            one(|out| {
                sl_solar_corona_delay(
                    &raw const tx,
                    &raw const rx,
                    8.4e9,
                    false,
                    core::ptr::null(),
                    0,
                    out,
                )
            })
        };
        assert_eq!(phase, -delay);

        // The caller's own series is used instead of the default, so a binding that ignored
        // `series` would answer `delay` here.
        let mine = [SlCoronaTerm {
            coefficient: 1.0e12,
            exponent: 2.0,
        }];
        let single = [CoronaTerm {
            coefficient: 1.0e12,
            exponent: 2.0,
        }];
        // SAFETY: `mine` is a live local array of exactly one element.
        let custom = unsafe {
            one(|out| {
                sl_solar_corona_delay(
                    &raw const tx,
                    &raw const rx,
                    8.4e9,
                    true,
                    mine.as_ptr(),
                    mine.len(),
                    out,
                )
            })
        };
        assert_eq!(
            custom,
            estimation_solar_corona_delay(
                &CoronaSeries::new(&single).expect("one term is a series"),
                &to_three(tx),
                &to_three(rx),
                8.4e9,
                PlasmaSignature::GroupDelay
            )
            .expect("the ray clears the photosphere")
        );
        assert!(custom != delay, "the caller's series was ignored");

        // A ray through the photosphere is refused, because there is no honest correction for it.
        let through = SlVec3 {
            x: -1.5e11,
            y: 0.0,
            z: 0.0,
        };
        let past = SlVec3 {
            x: 2.3e11,
            y: 0.0,
            z: 0.0,
        };
        let mut slot = 0.0;
        // SAFETY: all three pointers are to live locals.
        let status = unsafe {
            sl_solar_corona_delay(
                &raw const through,
                &raw const past,
                8.4e9,
                true,
                core::ptr::null(),
                0,
                &raw mut slot,
            )
        };
        assert_eq!(status, SL_ERR_DOMAIN);

        // Shapiro: equal to its crate function, symmetric in the endpoints, and exactly linear in
        // the central mass — which together pin the logarithm's argument without restating it.
        let (r1, r2, path) = (6.878e6, 2.6561e7, 2.0e7);
        let leg = sl_shapiro_delay(r1, r2, path, MU_EARTH);
        assert_eq!(leg, estimation_shapiro_delay(r1, r2, path, MU_EARTH));
        assert_eq!(leg, sl_shapiro_delay(r2, r1, path, MU_EARTH));
        assert_eq!(sl_shapiro_delay(r1, r2, path, 2.0 * MU_EARTH), 2.0 * leg);
        assert!(
            (0.001..0.05).contains(&leg),
            "{leg} m is not a millimetre-scale delay",
        );
    }

    /// A cannonball at `altitude` under an exponential atmosphere pinned at 300 km, with the
    /// defaults `sl_decay_params_init` writes and the four an atmosphere cannot be guessed for.
    ///
    /// Collected once because all three lifetime entry points take the same thirteen numbers, and a
    /// test that retyped them would be testing its own transcription.
    fn decay_case(altitude: f64) -> (SlDecayParams, [f64; 6], [f64; 3]) {
        let mut p = SlDecayParams {
            cd: f64::NAN,
            area_over_mass: f64::NAN,
            reference_density: f64::NAN,
            reference_radius: f64::NAN,
            scale_height: f64::NAN,
            horizon: f64::NAN,
            chunk: f64::NAN,
            tolerance: f64::NAN,
            density_scale: f64::NAN,
            mu: f64::NAN,
            body_radius: f64::NAN,
            reentry_altitude: f64::NAN,
            rotation_rate: f64::NAN,
        };
        // SAFETY: `p` is a live local of exactly this type.
        unsafe { sl_decay_params_init(&raw mut p) };
        p.area_over_mass = 0.02;
        p.reference_density = 2e-11;
        p.reference_radius = RE_EARTH + 300e3;
        p.scale_height = 60e3;
        p.horizon = 400.0 * 86_400.0;
        p.tolerance = 3_600.0;
        (
            p,
            [RE_EARTH + altitude, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -J2_EARTH],
        )
    }

    /// Run one decay case, or fail naming the refusal.
    fn decayed(p: SlDecayParams, initial: [f64; 6], zonals: [f64; 3]) -> SlLifetime {
        let mut out = SlLifetime {
            outcome: -1,
            seconds: f64::NAN,
            perigee_altitude_m: f64::NAN,
        };
        // SAFETY: every pointer is to a live local of the right type and extent.
        let status = unsafe {
            sl_orbital_lifetime(
                &raw const p,
                initial.as_ptr(),
                zonals.as_ptr(),
                zonals.len(),
                &raw mut out,
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        out
    }

    /// Re-entry, survival and breakdown are three answers, and the binding keeps them apart.
    #[test]
    fn the_lifetime_bindings_separate_their_outcomes_and_follow_the_air() {
        let (params, initial, zonals) = decay_case(300e3);
        let low = decayed(params, initial, zonals);
        assert_eq!(low.outcome, LIFETIME_REENTRY);
        assert!(low.seconds > 0.0 && low.seconds < params.horizon, "{low:?}");
        assert!(
            low.perigee_altitude_m <= params.reentry_altitude + 1e3,
            "reported at {} m, threshold {}",
            low.perigee_altitude_m,
            params.reentry_altitude
        );

        // The same object 700 km higher flies through `exp(-700/60)` of the density and outlives
        // the horizon — and the second field is then the horizon itself rather than a decay epoch
        // that was never observed.
        let (high_params, high_initial, _) = decay_case(1000e3);
        let high = decayed(high_params, high_initial, zonals);
        assert_eq!(high.outcome, LIFETIME_SURVIVES);
        assert_eq!(high.seconds, high_params.horizon);

        // The density scale is the ensemble's own axis, and it acts the way the physics says.
        let mut denser = params;
        denser.density_scale = 2.0;
        let sooner = decayed(denser, initial, zonals);
        assert!(
            sooner.seconds < low.seconds,
            "{} is not sooner than {}",
            sooner.seconds,
            low.seconds
        );

        // The shell co-rotates, so the satellite meets it at `v − ω×r` rather than at `v` -- some
        // 480 m/s of the 7.7 km/s at this radius. Freezing it is the faster headwind and must bring
        // the object down sooner; a lifetime that did not move would mean `rotation_rate` never
        // reached the drag quadrature, which no assertion on the atmosphere alone could see.
        let mut frozen = params;
        frozen.rotation_rate = 0.0;
        assert!(
            decayed(frozen, initial, zonals).seconds < low.seconds,
            "a frozen atmosphere must bring it down sooner than a co-rotating one"
        );

        // A three-member ensemble's quantiles are its three members, in order.
        let scales = [0.5, 1.0, 2.0];
        let mut members: Vec<f64> = scales
            .iter()
            .map(|&s| {
                let mut p = params;
                p.density_scale = s;
                decayed(p, initial, zonals).seconds
            })
            .collect();
        members.sort_by(f64::total_cmp);
        let quantiles = [0.0, 0.5, 1.0];
        let mut seconds = [f64::NAN; 3];
        let mut ensemble = SlLifetimeEnsemble {
            members: 0,
            survivors: 0,
            unresolved: 0,
        };
        // SAFETY: every pointer is to a live local of the stated extent.
        let status = unsafe {
            sl_orbital_lifetime_ensemble(
                &raw const params,
                initial.as_ptr(),
                zonals.as_ptr(),
                zonals.len(),
                scales.as_ptr(),
                scales.len(),
                quantiles.as_ptr(),
                quantiles.len(),
                seconds.as_mut_ptr(),
                &raw mut ensemble,
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert_eq!(
            (ensemble.members, ensemble.survivors, ensemble.unresolved),
            (3, 0, 0)
        );
        assert_eq!(seconds[0], members[0]);
        assert_eq!(seconds[2], members[2]);
        assert!(
            seconds[0] <= seconds[1] && seconds[1] <= seconds[2],
            "{seconds:?}"
        );

        // A quantile past the censoring fraction is NaN rather than the largest decay the run
        // happened to see, which would understate the answer by however long the censored member
        // goes on to last. One survivor in three censors the top of the sample.
        let censoring = [0.5, 1.0, 1e-3];
        let mut censored = [f64::NAN; 3];
        // SAFETY: as above.
        let status = unsafe {
            sl_orbital_lifetime_ensemble(
                &raw const params,
                initial.as_ptr(),
                zonals.as_ptr(),
                zonals.len(),
                censoring.as_ptr(),
                censoring.len(),
                quantiles.as_ptr(),
                quantiles.len(),
                censored.as_mut_ptr(),
                &raw mut ensemble,
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert_eq!((ensemble.members, ensemble.survivors), (3, 1));
        assert!(censored[0].is_finite(), "{censored:?}");
        assert!(censored[2].is_nan(), "{censored:?}");

        // `T ∝ 1/(S/m)` to first order, so the log-log slope is −1 and its sign is the whole point.
        // SAFETY: every pointer is to a live local of the right type and extent.
        let slope = unsafe {
            one(|out| {
                sl_ballistic_log_sensitivity(
                    &raw const params,
                    initial.as_ptr(),
                    zonals.as_ptr(),
                    zonals.len(),
                    0.05,
                    out,
                )
            })
        };
        assert!(
            (slope + 1.0).abs() < 0.2,
            "d ln T / d ln (S/m) = {slope}, expected about -1"
        );

        // A zonal list that stops before degree two is refused here rather than asserted on inside
        // `DsstZonal`, where an assertion across the ABI is an abort and not a status.
        let short = [0.0, 0.0];
        let mut out = SlLifetime {
            outcome: -1,
            seconds: f64::NAN,
            perigee_altitude_m: f64::NAN,
        };
        // SAFETY: `short` is a live local array of exactly two elements.
        let status = unsafe {
            sl_orbital_lifetime(
                &raw const params,
                initial.as_ptr(),
                short.as_ptr(),
                short.len(),
                &raw mut out,
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
        assert!(last_error().contains("zonal_count"), "{}", last_error());

        // `_init` leaves the four an atmosphere cannot be guessed for as NaN, so a caller that
        // takes the defaults and fills nothing in is refused rather than run on a silent zero.
        let mut bare = params;
        bare.reference_density = f64::NAN;
        // SAFETY: as above.
        let status = unsafe {
            sl_orbital_lifetime(
                &raw const bare,
                initial.as_ptr(),
                zonals.as_ptr(),
                zonals.len(),
                &raw mut out,
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
        assert!(
            last_error().contains("reference_density"),
            "{}",
            last_error()
        );
    }

    /// The exponential shell on its own, rather than through a decade of quadrature.
    ///
    /// Every lifetime case above is an equatorial circular orbit -- `sl_orbital_lifetime` takes a
    /// mean *equinoctial* state, and `[a, 0, 0, 0, 0, 0]` is one -- so `z` is zero at every node and
    /// the co-rotation never leaves the orbit plane. A dropped `z²` and a reversed rotation are both
    /// invisible from an outcome and a decay epoch, and a decade of drag absorbs either.
    #[test]
    fn the_decay_atmosphere_is_the_exponential_shell_it_documents() {
        let omega = 7.292_115e-5;
        let air = CapiDecayAtmosphere {
            reference_density: 2e-11,
            reference_radius: 6_678_137.0,
            scale_height: 60e3,
            omega,
            scale: 1.5,
        };

        // Off the equator, and with the three components distinct, so the radius is neither the
        // horizontal distance nor any pairwise combination of them.
        let position = [3_000e3, 4_000e3, 5_000e3];
        let radius = 50e12_f64.sqrt();
        let expected = 1.5 * 2e-11 * ((6_678_137.0 - radius) / 60e3).exp();
        let got = air.density(0.0, position);
        assert!(
            (got - expected).abs() < 1e-12 * expected,
            "{got} against {expected}"
        );
        // The two parameters the shell is named by: at the reference radius the density is the
        // scaled reference density exactly, and one scale height out it has fallen by `e`.
        let at_reference = air.density(0.0, [6_678_137.0, 0.0, 0.0]);
        assert!(
            (at_reference - 1.5 * 2e-11).abs() < 1e-25,
            "at the reference radius the density is {at_reference}"
        );
        let one_height_out = air.density(0.0, [0.0, 0.0, 6_678_137.0 + 60e3]);
        assert!(
            (one_height_out * core::f64::consts::E - at_reference).abs() < 1e-25,
            "a scale height out the density is {one_height_out}"
        );

        // `ω ẑ × r`, which is `[-ω y, ω x, 0]` and nothing else: horizontal, perpendicular to the
        // horizontal projection of the position, of magnitude `ω √(x² + y²)`, and prograde. A
        // reflection satisfies the first three and not the fourth, so the vector is pinned as
        // written rather than as a set of properties.
        let wind = air.velocity(position);
        assert_eq!(wind, [-omega * 4_000e3, omega * 3_000e3, 0.0]);
        let dot = wind[0].mul_add(position[0], wind[1] * position[1]);
        let term = (wind[0] * position[0]).abs();
        assert!(dot.abs() < 1e-15 * term, "r·(ω ẑ × r) is {dot}, not 0");
        assert!(
            (wind[0].hypot(wind[1]) - omega * 5_000e3).abs() < 1e-9,
            "the wind speed is {}",
            wind[0].hypot(wind[1])
        );
        // The shell co-rotates rigidly, so the wind does not read the height it is asked at.
        assert_eq!(air.velocity([3_000e3, 4_000e3, -9_000e3]), wind);
    }

    /// The TLE's modulo-10 checksum: digits count for themselves, minus signs for one.
    fn tle_checksum(line: &str) -> u32 {
        line.bytes()
            .take(68)
            .map(|c| match c {
                b'0'..=b'9' => u32::from(c - b'0'),
                b'-' => 1,
                _ => 0,
            })
            .sum::<u32>()
            % 10
    }

    /// The TLE binding puts every field in its own columns, and both checksums are correct.
    ///
    /// The lines are pinned literally. A column layout has no tolerance to assert to: a field one
    /// column out still parses, still checksums, and is a different orbit.
    #[test]
    fn the_tle_binding_emits_both_lines_in_their_own_columns() {
        let epoch = SlUtcDateTime {
            year: 2008,
            month: 9,
            day: 20,
            hour: 12,
            minute: 25,
            second: 12.25,
        };
        let parameters = [
            15.721_253_91,
            0.000_670_3,
            51.641_6,
            247.462_7,
            130.536_0,
            325.028_8,
        ];
        let designator = c"1998-067A";
        let mut out = SlTle {
            line1: [0; TLE_LINE_BYTES],
            line2: [0; TLE_LINE_BYTES],
        };
        // SAFETY: every pointer is to a live local; the designator is a NUL-terminated literal.
        let status = unsafe {
            sl_tle_lines(
                &raw const epoch,
                25544,
                parameters.as_ptr(),
                -1.1606e-4,
                designator.as_ptr(),
                292,
                56353,
                &raw mut out,
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());

        let read = |field: &[c_char; TLE_LINE_BYTES]| {
            let bytes: Vec<u8> = field
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| u8::try_from(c).expect("a TLE line is ASCII"))
                .collect();
            String::from_utf8(bytes).expect("a TLE line is ASCII")
        };
        let (line1, line2) = (read(&out.line1), read(&out.line2));
        assert_eq!(
            line1,
            "1 25544U 98067A   08264.51750289  .00000000  00000-0 -11606-3 0  2921"
        );
        assert_eq!(
            line2,
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        );
        // The checksums are recomputed rather than read out of the literals, so a transcription
        // slip in the two strings above is caught here instead of being pinned by them.
        for line in [&line1, &line2] {
            assert_eq!(line.len(), 69, "{line}");
            assert_eq!(
                u32::from(line.as_bytes()[68] - b'0'),
                tle_checksum(line),
                "{line}"
            );
        }
        // Both fields are NUL-terminated in their last byte, which is what a C caller reads them
        // as.
        assert_eq!(out.line1[TLE_LINE_BYTES - 1], 0);
        assert_eq!(out.line2[TLE_LINE_BYTES - 1], 0);

        // A catalog number past the five numeric columns is refused rather than truncated or
        // Alpha-5 encoded, which is why it is checked before the line is built.
        // SAFETY: as above.
        let status = unsafe {
            sl_tle_lines(
                &raw const epoch,
                270_123,
                parameters.as_ptr(),
                0.0,
                core::ptr::null(),
                0,
                0,
                &raw mut out,
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
    }

    /// The dual-frequency combinations, against their crate functions and against the biases they
    /// were built from.
    #[test]
    fn the_carrier_combination_bindings_recover_what_was_injected() {
        let (f1, f2) = (1_575.42e6, 1_227.60e6);
        let pair = FrequencyPair::new(f1, f2).expect("L1 is above L2");
        let mut lanes = SlLanes {
            lambda1: 0.0,
            lambda2: 0.0,
            wide_lane: 0.0,
            narrow_lane: 0.0,
        };
        // SAFETY: `lanes` is a live local of exactly this type.
        assert_eq!(
            unsafe { sl_lane_wavelengths(f1, f2, &raw mut lanes) },
            SL_OK,
            "{}",
            last_error()
        );
        assert_eq!(lanes.lambda1, pair.wavelength1());
        assert_eq!(lanes.lambda2, pair.wavelength2());
        assert_eq!(lanes.wide_lane, pair.wide_lane_wavelength());
        assert_eq!(lanes.narrow_lane, pair.narrow_lane_wavelength());
        // The published GPS L1/L2 lanes: the wide one is why its ambiguity is resolved first.
        assert!((lanes.wide_lane - 0.8619).abs() < 1e-4, "{lanes:?}");
        assert!((lanes.narrow_lane - 0.1070).abs() < 1e-4, "{lanes:?}");

        // The RINEX letters, and the FDMA hole that is a documented answer rather than an omission.
        assert_eq!(sl_carrier_frequency(b'G' as c_char, b'1' as c_char), f1);
        assert_eq!(sl_carrier_frequency(b'G' as c_char, b'2' as c_char), f2);
        assert!(sl_carrier_frequency(b'R' as c_char, b'1' as c_char).is_nan());
        assert!(sl_carrier_frequency(b'?' as c_char, b'1' as c_char).is_nan());

        // A geometric range with a first-order ionospheric delay on each frequency, and the two
        // carriers' own integer ambiguities.
        let (geometric, i1) = (22_100_000.0, 4.5);
        let i2 = i1 * (f1 / f2) * (f1 / f2);
        let (p1, p2) = (geometric + i1, geometric + i2);
        let (n1, n2) = (1_234.0, 987.0);
        let l1 = lanes.lambda1.mul_add(n1, geometric - i1);
        let l2 = lanes.lambda2.mul_add(n2, geometric - i2);
        // SAFETY: `out` in each closure is a live local of `one`.
        unsafe {
            let free = one(|out| sl_ionosphere_free(f1, f2, p1, p2, out));
            assert_eq!(free, pair.ionosphere_free(p1, p2));
            // The combination exists to remove exactly this, and it removes it to the last micron.
            assert!(
                (free - geometric).abs() < 1e-6,
                "{free} against {geometric}"
            );

            let wl = one(|out| sl_wide_lane(f1, f2, p1, p2, out));
            assert_eq!(wl, pair.wide_lane(p1, p2));
            let nl = one(|out| sl_narrow_lane(f1, f2, p1, p2, out));
            assert_eq!(nl, pair.narrow_lane(p1, p2));
            // On code the two lanes carry the ionosphere with equal magnitude `A/(f1·f2)` and
            // opposite sign, so their sum is exactly twice the geometric range and they sit either
            // side of it. That pins both combinations against each other rather than one at a time.
            assert!(wl < geometric && geometric < nl, "{wl} {geometric} {nl}");
            assert!(
                (wl + nl - 2.0 * geometric).abs() < 1e-6,
                "{wl} and {nl} do not straddle {geometric}"
            );

            let cycles = one(|out| sl_melbourne_wubbena(f1, f2, p1, p2, l1, l2, true, out));
            assert_eq!(cycles, pair.melbourne_wubbena_cycles(p1, p2, l1, l2));
            // Everything geometric and everything dispersive cancels, so what is left is `N1 - N2`.
            assert!((cycles - (n1 - n2)).abs() < 1e-6, "{cycles}");
            let metres = one(|out| sl_melbourne_wubbena(f1, f2, p1, p2, l1, l2, false, out));
            assert_eq!(metres, pair.melbourne_wubbena(p1, p2, l1, l2));
            assert!(
                (metres - cycles * lanes.wide_lane).abs() < 1e-6,
                "{metres} is not {cycles} wide-lane cycles"
            );
        }

        assert_eq!(sl_geometry_free(p1, p2), gnss_geometry_free(p1, p2));
        assert!((sl_geometry_free(p1, p2) - (i1 - i2)).abs() < 1e-8);
        assert_eq!(sl_graphic(p1, l1), gnss_graphic(p1, l1));

        // Two equal frequencies have no lane at all, and a swapped pair has a negative one; both
        // are refusals rather than an infinity and a sign-flipped ambiguity.
        let mut slot = 0.0;
        // SAFETY: both out-pointers are to live locals.
        unsafe {
            assert_eq!(
                sl_ionosphere_free(f1, f1, p1, p2, &raw mut slot),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_lane_wavelengths(f2, f1, &raw mut lanes),
                SL_ERR_INVALID_ARGUMENT
            );
        }
    }

    /// Each detector reports the verdict it can reach, and the two reach different ones.
    #[test]
    fn the_cycle_slip_bindings_report_the_verdict_each_detector_can_reach() {
        // A geometry-free series drifting 0.02 m per 30 s sample with a 0.30 m step at index 20: a
        // slip a 0.10 m first-difference threshold sees and a 0.02 m drift does not trip.
        let count = 40_usize;
        let times: Vec<f64> = (0..count).map(|i| 30.0 * i as f64).collect();
        let values: Vec<f64> = (0..count)
            .map(|i| {
                let drift = 0.02 * i as f64;
                if i >= 20 { drift + 0.30 } else { drift }
            })
            .collect();
        let mut verdicts = vec![
            SlSlipVerdict {
                verdict: -1,
                statistic: 0.0,
            };
            count
        ];
        // SAFETY: all three pointers are to live buffers of exactly `count` elements.
        let status = unsafe {
            sl_detect_cycle_slips_geometry_free(
                times.as_ptr(),
                values.as_ptr(),
                count,
                0.10,
                120.0,
                verdicts.as_mut_ptr(),
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert_eq!(verdicts[0].verdict, SLIP_ARC_START);
        // Nothing was tested at the arc start, and the statistic says so rather than reading zero,
        // which is the value a perfectly continuous sample would have.
        assert!(verdicts[0].statistic.is_nan(), "{:?}", verdicts[0]);
        assert_eq!(verdicts[19].verdict, SLIP_CONTINUOUS);
        assert_eq!(verdicts[20].verdict, SLIP_SLIPPED);
        assert!(
            (verdicts[20].statistic - 0.32).abs() < 1e-12,
            "{:?}",
            verdicts[20]
        );
        assert_eq!(verdicts[21].verdict, SLIP_CONTINUOUS);
        assert_eq!(
            verdicts
                .iter()
                .filter(|v| v.verdict == SLIP_SLIPPED)
                .count(),
            1
        );

        // The polynomial detector tests nothing until its window fills, and clears the window after
        // a slip — so the two detectors disagree on the sample after the step, and that disagreement
        // is why there are two entry points.
        let mut fitted = verdicts.clone();
        // SAFETY: as above.
        let status = unsafe {
            sl_detect_cycle_slips_polynomial(
                times.as_ptr(),
                values.as_ptr(),
                count,
                1,
                5,
                0.10,
                120.0,
                fitted.as_mut_ptr(),
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert!(
            fitted[..5].iter().all(|v| v.verdict == SLIP_ARC_START),
            "{:?}",
            &fitted[..5]
        );
        assert_eq!(fitted[5].verdict, SLIP_CONTINUOUS);
        assert_eq!(fitted[20].verdict, SLIP_SLIPPED);
        assert!(
            (fitted[20].statistic - 0.30).abs() < 1e-9,
            "{:?}",
            fitted[20]
        );
        assert_eq!(fitted[21].verdict, SLIP_ARC_START);

        // A window that cannot hold the fit is refused: `window < order + 2` leaves no residual.
        // SAFETY: as above.
        let status = unsafe {
            sl_detect_cycle_slips_polynomial(
                times.as_ptr(),
                values.as_ptr(),
                count,
                3,
                4,
                0.10,
                120.0,
                fitted.as_mut_ptr(),
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
        // A threshold that is not positive is the other half of the detectors' own contract.
        // SAFETY: as above.
        let status = unsafe {
            sl_detect_cycle_slips_geometry_free(
                times.as_ptr(),
                values.as_ptr(),
                count,
                0.0,
                120.0,
                verdicts.as_mut_ptr(),
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
    }

    /// The ambiguity bindings answer in the caller's own parametrisation, not the decorrelated one.
    #[test]
    fn the_ambiguity_bindings_return_candidates_in_the_callers_parametrisation() {
        // A correlated three-dimensional problem, so the decorrelation has something to do and `Z`
        // is not the identity.
        let float = [5.45_f64, 3.10, 2.97];
        let covariance = [
            6.290, 5.978, 0.544, //
            5.978, 6.292, 2.340, //
            0.544, 2.340, 6.288,
        ];
        let reduced = Reduction::new(&float, &covariance).expect("a covariance is a covariance");

        let mut z = [0.0_f64; 9];
        let mut d = [0.0_f64; 3];
        // SAFETY: every pointer is to a live local of the stated extent.
        let status = unsafe {
            sl_ambiguity_decorrelation(
                float.as_ptr(),
                float.len(),
                covariance.as_ptr(),
                z.as_mut_ptr(),
                d.as_mut_ptr(),
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert_eq!(z.as_slice(), reduced.transformation());
        assert_eq!(d.as_slice(), reduced.conditional_variances());
        // `Z` is unimodular, so every entry is an integer — the one property of the transformation
        // that survives any correct reduction and no incorrect one.
        assert!(z.iter().all(|v| (v - v.round()).abs() < 1e-9), "{z:?}");
        assert!(d.iter().all(|&v| v > 0.0), "{d:?}");

        let mut ambiguities = [0_i64; 3];
        let mut norm = f64::NAN;
        // SAFETY: as above.
        let status = unsafe {
            sl_ambiguity_bootstrap(
                float.as_ptr(),
                float.len(),
                covariance.as_ptr(),
                ambiguities.as_mut_ptr(),
                &raw mut norm,
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        let expected = Reduction::new(&float, &covariance)
            .expect("a covariance is a covariance")
            .bootstrap();
        assert_eq!(ambiguities.as_slice(), expected.ambiguities.as_slice());
        assert_eq!(norm, expected.squared_norm);

        let mut best = [0_i64; 6];
        let mut norms = [f64::NAN; 2];
        let mut len = 0_usize;
        // SAFETY: `best` holds `count * n` and `norms` holds `count`, which is what is passed.
        let status = unsafe {
            sl_resolve_ambiguities(
                float.as_ptr(),
                float.len(),
                covariance.as_ptr(),
                2,
                best.as_mut_ptr(),
                norms.as_mut_ptr(),
                &raw mut len,
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert_eq!(len, 2);
        let searched = Reduction::new(&float, &covariance)
            .expect("a covariance is a covariance")
            .search(2)
            .expect("the ellipsoid holds two candidates");
        assert_eq!(&best[..3], searched[0].ambiguities.as_slice());
        assert_eq!(&best[3..], searched[1].ambiguities.as_slice());
        assert_eq!(norms[0], searched[0].squared_norm);
        assert_eq!(norms[1], searched[1].squared_norm);
        assert!(norms[0] <= norms[1], "{norms:?}");
        // The search is at least as good as the bootstrap, which is the whole reason it exists.
        assert!(norms[0] <= norm, "{} against {norm}", norms[0]);

        // A covariance that is not one is refused before anything is factorised.
        let broken = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0];
        // SAFETY: as above.
        let status = unsafe {
            sl_ambiguity_decorrelation(
                float.as_ptr(),
                float.len(),
                broken.as_ptr(),
                z.as_mut_ptr(),
                d.as_mut_ptr(),
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
        // So is an empty problem, which would otherwise index an empty factorisation.
        // SAFETY: as above.
        let status = unsafe {
            sl_ambiguity_decorrelation(
                float.as_ptr(),
                0,
                covariance.as_ptr(),
                z.as_mut_ptr(),
                d.as_mut_ptr(),
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
    }

    /// A head-on conjunction with isotropic covariances, the miss vector across the relative track.
    fn conjunction_case(miss: f64) -> SlConjunction {
        let isotropic = [1e4, 0.0, 0.0, 0.0, 1e4, 0.0, 0.0, 0.0, 1e4];
        SlConjunction {
            relative_position: SlVec3 {
                x: 0.6 * miss,
                y: 0.0,
                z: 0.8 * miss,
            },
            relative_velocity: SlVec3 {
                x: 0.0,
                y: 14_000.0,
                z: 0.0,
            },
            primary_covariance: isotropic,
            secondary_covariance: isotropic,
            primary_radius: 5.0,
            secondary_radius: 5.0,
        }
    }

    /// A `conjunction_case` propagated rectilinearly with the joint covariance scaled by `scale` —
    /// which is exactly the motion the 2-D model assumes when the scale is one.
    #[repr(C)]
    struct RectilinearCase {
        geometry: SlConjunction,
        scale: f64,
    }

    unsafe extern "C" fn rectilinear_sample(
        user_data: *mut c_void,
        t: f64,
        out: *mut SlRelativeSample,
    ) -> i32 {
        // SAFETY: the test passes a live `RectilinearCase`, and `out` is the library's own slot.
        unsafe {
            let case = &*user_data.cast::<RectilinearCase>();
            let g = case.geometry;
            let mut joint = [0.0_f64; 9];
            for (slot, (a, b)) in joint.iter_mut().zip(
                g.primary_covariance
                    .iter()
                    .zip(g.secondary_covariance.iter()),
            ) {
                *slot = case.scale * (a + b);
            }
            *out = SlRelativeSample {
                position: SlVec3 {
                    x: g.relative_velocity.x.mul_add(t, g.relative_position.x),
                    y: g.relative_velocity.y.mul_add(t, g.relative_position.y),
                    z: g.relative_velocity.z.mul_add(t, g.relative_position.z),
                },
                velocity: g.relative_velocity,
                covariance: joint,
            };
        }
        SL_OK
    }

    /// A sampler that cannot produce a state, which is what ends a scan honestly.
    unsafe extern "C" fn refusing_sample(_: *mut c_void, _: f64, _: *mut SlRelativeSample) -> i32 {
        SL_ERR_COMPUTE
    }

    /// Amplitude of the along-track excursion the screening test scans, metres.
    const DRIFT_AMPLITUDE: f64 = 10_000.0;
    /// Its period, seconds — so the minima are 300 s apart.
    const DRIFT_PERIOD: f64 = 600.0;
    /// The constant across-track separation, metres.
    const DRIFT_MISS: f64 = 500.0;

    /// A relative trajectory whose closest approaches are known in closed form: along-track the
    /// separation is `A·sin(2πt/P)` and across-track a constant `miss`, so the minima sit where the
    /// sine crosses zero, each at `miss` and at speed `A·2π/P`. Nothing is read back from the scan.
    unsafe extern "C" fn drifting_state(_: *mut c_void, t: f64, out: *mut SlState) -> i32 {
        let rate = 2.0 * core::f64::consts::PI / DRIFT_PERIOD;
        // SAFETY: `out` is the library's own slot.
        unsafe {
            *out = SlState {
                position: SlVec3 {
                    x: DRIFT_AMPLITUDE * (rate * t).sin(),
                    y: DRIFT_MISS,
                    z: 0.0,
                },
                velocity: SlVec3 {
                    x: DRIFT_AMPLITUDE * rate * (rate * t).cos(),
                    y: 0.0,
                    z: 0.0,
                },
            };
        }
        SL_OK
    }

    unsafe extern "C" fn refusing_state(_: *mut c_void, _: f64, _: *mut SlState) -> i32 {
        SL_ERR_COMPUTE
    }

    /// The 2-D probability, by all four estimators, against the encounter plane it is taken on.
    #[test]
    fn the_collision_probability_binding_is_the_encounter_plane_integral() {
        let geometry = conjunction_case(50.0);
        let plane = to_conjunction(geometry)
            .project()
            .expect("the encounter velocity is not degenerate");
        let mut estimates = SlCollisionProbability {
            foster: f64::NAN,
            patera: f64::NAN,
            chan: f64::NAN,
            maximum: f64::NAN,
        };
        // SAFETY: both pointers are to live locals of the right type.
        let status = unsafe { sl_collision_probability(&raw const geometry, &raw mut estimates) };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert_eq!(estimates.foster, plane.probability_foster());
        assert_eq!(estimates.patera, plane.probability_patera());
        assert_eq!(estimates.chan, plane.probability_chan());
        assert_eq!(estimates.maximum, plane.probability_max());
        let reference = estimates.foster;

        // Three routes to one integral, so they agree; the fourth is a bound over a scaling of the
        // covariance rather than a fourth estimate of the same thing, so it sits above all three.
        for estimate in [estimates.patera, estimates.chan] {
            assert!(
                (estimate - reference).abs() < 0.02 * reference,
                "{estimates:?}"
            );
        }
        assert!(estimates.maximum >= reference, "{estimates:?}");

        // Head-on with an isotropic joint covariance `2σ²` and no miss, the integral is elementary:
        // `1 − exp(−R²/2s²)`. The one geometry here whose answer is not itself a quadrature.
        let centred = conjunction_case(0.0);
        let exact = 1.0 - (-(10.0_f64 * 10.0) / (2.0 * 2.0 * 100.0 * 100.0)).exp();
        // SAFETY: as above.
        let status = unsafe { sl_collision_probability(&raw const centred, &raw mut estimates) };
        assert_eq!(status, SL_OK, "{}", last_error());
        for estimate in [estimates.foster, estimates.patera, estimates.chan] {
            assert!(
                (estimate - exact).abs() < 2e-4 * exact,
                "{estimate} against {exact}"
            );
        }

        // Further away is less likely, which is the one monotonicity the whole model rests on.
        let far = conjunction_case(300.0);
        // SAFETY: as above.
        let status = unsafe { sl_collision_probability(&raw const far, &raw mut estimates) };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert!(estimates.foster < reference, "{estimates:?}");

        // A degenerate encounter velocity has no encounter plane, and that is a domain refusal
        // rather than a probability of zero.
        let mut still = conjunction_case(50.0);
        still.relative_velocity = SlVec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        // SAFETY: as above.
        let status = unsafe { sl_collision_probability(&raw const still, &raw mut estimates) };
        assert_eq!(status, SL_ERR_DOMAIN);
    }

    /// The domain check clears the encounter it was built from, and the period is its scale.
    #[test]
    fn the_domain_check_binding_clears_its_own_assumption_and_scales_with_the_period() {
        let geometry = conjunction_case(50.0);
        let mut case = RectilinearCase {
            geometry,
            scale: 1.0,
        };
        let assess = |case: &mut RectilinearCase, period: f64| {
            let mut out = SlUsageViolations {
                any: true,
                non_positive_definite: true,
                offset: true,
                extended: true,
                inaccuracy: true,
                offset_indicator: f64::NAN,
                extended_indicator: f64::NAN,
                inaccuracy_indicator: f64::NAN,
                correction_factor: f64::NAN,
            };
            // SAFETY: `case` outlives the call and the sampler reads only through that pointer;
            // `out` is a live local.
            let status = unsafe {
                sl_conjunction_usage_violations(
                    &raw const geometry,
                    period,
                    Some(rectilinear_sample),
                    core::ptr::from_mut(case).cast::<c_void>(),
                    &raw mut out,
                )
            };
            assert_eq!(status, SL_OK, "{}", last_error());
            out
        };

        let period = 5_400.0;
        let clear = assess(&mut case, period);
        // The sampler moves exactly the way the 2-D model assumes, so nothing is out of domain, the
        // disagreement is the bisection's own width and the correction it implies is unity.
        assert!(
            !(clear.any
                || clear.non_positive_definite
                || clear.offset
                || clear.extended
                || clear.inaccuracy),
            "{clear:?}"
        );
        assert!((clear.correction_factor - 1.0).abs() < 1e-6, "{clear:?}");
        assert!(clear.inaccuracy_indicator < 1e-6, "{clear:?}");

        // The period is the scale both ratios are stated on, and halving it doubles them exactly —
        // which is what says it entered where the thresholds expect it and not somewhere else.
        let halved = assess(&mut case, 0.5 * period);
        assert!(
            (halved.offset_indicator - 2.0 * clear.offset_indicator).abs() < 1e-9,
            "{halved:?}"
        );
        assert!(
            (halved.extended_indicator - 2.0 * clear.extended_indicator).abs() < 1e-9,
            "{halved:?}"
        );

        // A short enough period puts the encounter past the published extended-duration limit, and
        // the flag follows the indicator across the threshold rather than sitting beside it.
        let short = clear.extended_indicator * period / 0.021;
        let flagged = assess(&mut case, short);
        assert!(
            flagged.extended_indicator > 0.02 && flagged.extended && flagged.any,
            "{flagged:?}"
        );

        // A covariance that grows away from TCA is a real disagreement between the two encounters,
        // and the correction factor is what reports it.
        case.scale = 4.0;
        let grown = assess(&mut case, period);
        assert!(grown.inaccuracy_indicator > 1e-3, "{grown:?}");
        assert!((grown.correction_factor - 1.0).abs() > 1e-3, "{grown:?}");
        case.scale = 1.0;

        // A sampler that refuses is reported as a refusal, not as a check that found nothing wrong.
        let mut out = SlUsageViolations {
            any: false,
            non_positive_definite: false,
            offset: false,
            extended: false,
            inaccuracy: false,
            offset_indicator: 0.0,
            extended_indicator: 0.0,
            inaccuracy_indicator: 0.0,
            correction_factor: 0.0,
        };
        // SAFETY: the callback reads nothing; `out` is a live local.
        let status = unsafe {
            sl_conjunction_usage_violations(
                &raw const geometry,
                period,
                Some(refusing_sample),
                core::ptr::null_mut(),
                &raw mut out,
            )
        };
        assert_eq!(status, SL_ERR_COMPUTE);
        assert!(
            last_error().contains("the sampler refused"),
            "{}",
            last_error()
        );

        // A period that is not one is refused by the model rather than rescaling both thresholds by
        // a number nobody chose.
        // SAFETY: `case` and `out` are live locals.
        let status = unsafe {
            sl_conjunction_usage_violations(
                &raw const geometry,
                0.0,
                Some(rectilinear_sample),
                core::ptr::from_mut(&mut case).cast::<c_void>(),
                &raw mut out,
            )
        };
        assert_eq!(status, SL_ERR_DOMAIN);
    }

    /// The screen finds the approaches it was given, sizes before it fills, and says where it
    /// cannot look.
    #[test]
    fn the_screening_bindings_find_their_own_minima_and_report_a_short_buffer() {
        let mut found = [SlCloseApproach {
            tca: f64::NAN,
            distance: f64::NAN,
            relative_speed: f64::NAN,
        }; 4];
        let (mut sizing, mut len) = (0_usize, 0_usize);
        // SAFETY: `out` is null on the sizing call, which is what the protocol documents; on the
        // second it is a live buffer of `found.len()`.
        unsafe {
            assert_eq!(
                sl_closest_approaches(
                    Some(drifting_state),
                    core::ptr::null_mut(),
                    10.0,
                    1_000.0,
                    30.0,
                    1e-4,
                    core::ptr::null_mut(),
                    0,
                    &raw mut sizing,
                ),
                SL_OK,
                "{}",
                last_error()
            );
            assert_eq!(
                sl_closest_approaches(
                    Some(drifting_state),
                    core::ptr::null_mut(),
                    10.0,
                    1_000.0,
                    30.0,
                    1e-4,
                    found.as_mut_ptr(),
                    found.len(),
                    &raw mut len,
                ),
                SL_OK,
                "{}",
                last_error()
            );
        }
        // The sizing call runs the same scan, so it has to agree with the filling one.
        assert_eq!(sizing, len);
        assert_eq!(len, 3);
        let rate = 2.0 * core::f64::consts::PI / DRIFT_PERIOD;
        for (approach, expected) in found[..len].iter().zip([300.0, 600.0, 900.0]) {
            assert!((approach.tca - expected).abs() < 0.5, "{approach:?}");
            assert!(
                (approach.distance - DRIFT_MISS).abs() < 1e-3,
                "{approach:?}"
            );
            assert!(
                (approach.relative_speed - DRIFT_AMPLITUDE * rate).abs() < 1e-3,
                "{approach:?}"
            );
        }

        // A buffer shorter than the scan found is a refusal that still reports the length, so the
        // caller can allocate and ask again rather than silently receiving two of three.
        let mut short = [found[0]; 2];
        let mut short_len = 0_usize;
        // SAFETY: `short` is a live buffer of exactly two elements.
        let status = unsafe {
            sl_closest_approaches(
                Some(drifting_state),
                core::ptr::null_mut(),
                10.0,
                1_000.0,
                30.0,
                1e-4,
                short.as_mut_ptr(),
                short.len(),
                &raw mut short_len,
            )
        };
        assert_eq!(status, SL_ERR_BUFFER_TOO_SMALL);
        assert_eq!(short_len, 3);

        // The step is a resolution as much as a bound: successive extrema here are 300 s apart, and
        // a step that cannot bracket them separately reports fewer approaches than there are. The
        // documented failure mode, pinned rather than assumed absent.
        let mut coarse = 0_usize;
        // SAFETY: sizing call, so `out` is null.
        let status = unsafe {
            sl_closest_approaches(
                Some(drifting_state),
                core::ptr::null_mut(),
                10.0,
                1_000.0,
                400.0,
                1e-4,
                core::ptr::null_mut(),
                0,
                &raw mut coarse,
            )
        };
        assert_eq!(status, SL_OK, "{}", last_error());
        assert!(coarse < len, "{coarse} against {len}");

        // A sampler that refuses ends the scan through the status rather than through an empty
        // answer, which is the difference between "nothing happened" and "we could not look".
        // SAFETY: `found` and `len` are live locals.
        let status = unsafe {
            sl_closest_approaches(
                Some(refusing_state),
                core::ptr::null_mut(),
                10.0,
                1_000.0,
                30.0,
                1e-4,
                found.as_mut_ptr(),
                found.len(),
                &raw mut len,
            )
        };
        assert_eq!(status, SL_ERR_COMPUTE);
        assert!(
            last_error().contains("the sampler refused"),
            "{}",
            last_error()
        );

        // An unusable scan request is refused by the scan itself, and each of these three is a
        // separate argument the caller can have got wrong.
        for (start, end, step, tolerance) in [
            (10.0, 1_000.0, -30.0, 1e-4),
            (1_000.0, 10.0, 30.0, 1e-4),
            (10.0, 1_000.0, 30.0, 0.0),
        ] {
            // SAFETY: sizing call, so `out` is null.
            let status = unsafe {
                sl_closest_approaches(
                    Some(drifting_state),
                    core::ptr::null_mut(),
                    start,
                    end,
                    step,
                    tolerance,
                    core::ptr::null_mut(),
                    0,
                    &raw mut coarse,
                )
            };
            assert_eq!(status, SL_ERR_INVALID_ARGUMENT, "{start} {end} {step}");
        }

        // The radial filter is filter 1 of an all-on-all screen: a certainty in one direction and
        // nothing claimed in the other.
        let low = (RE_EARTH + 400e3, RE_EARTH + 420e3);
        let high = (RE_EARTH + 800e3, RE_EARTH + 900e3);
        assert!(sl_radially_separated(low.0, low.1, high.0, high.1, 5_000.0));
        assert!(sl_radially_separated(high.0, high.1, low.0, low.1, 5_000.0));
        // The gap is 380 km, so a screening distance beyond it must not separate the pair.
        assert!(!sl_radially_separated(low.0, low.1, high.0, high.1, 400e3));
        // Overlapping shells are never separable, whatever the threshold.
        assert!(!sl_radially_separated(low.0, high.1, high.0, low.1, 1.0));
        // A corrupt element set is examined rather than silently discarded.
        assert!(!sl_radially_separated(
            f64::NAN,
            low.1,
            high.0,
            high.1,
            5_000.0
        ));
    }

    /// Every entry point this ticket added refuses a null out-pointer rather than writing through
    /// it.
    ///
    /// A missing null check is not a warning anywhere: it is a segmentation fault in the consumer's
    /// process, at whatever call site happened to pass an uninitialised pointer.
    #[test]
    fn every_new_fallible_entry_point_refuses_a_null_out_pointer() {
        let sat = leo();
        let tx = terminal_at(4_194_303.0, 172_460.0, 4_780_100.0, 0.0);
        let rx = terminal_at(-3_950_200.0, 2_522_400.0, 4_311_700.0, 0.04);
        let geometry = conjunction_case(50.0);
        let (params, initial, zonals) = decay_case(300e3);
        let segments = [SlRampSegment {
            start: 0.0,
            end: 1e4,
            start_frequency: 7.15e9,
            rate: 0.0,
        }];
        let link = SlDopplerLink {
            uplink: BAND_X,
            downlink: BAND_X,
            reference_frequency: 7.15e9,
            reference_band: BAND_X,
            count_interval: 60.0,
            subtracted: true,
        };
        let float = [0.1_f64, 0.2];
        let covariance = [1.0_f64, 0.0, 0.0, 1.0];
        let times = [0.0_f64, 30.0];
        let values = [0.0_f64, 0.01];
        let mut violations = SlUsageViolations {
            any: false,
            non_positive_definite: false,
            offset: false,
            extended: false,
            inaccuracy: false,
            offset_indicator: 0.0,
            extended_indicator: 0.0,
            inaccuracy_indicator: 0.0,
            correction_factor: 0.0,
        };
        let mut length = 0_usize;

        // SAFETY: every non-null pointer is to a live local of the right type and extent; the null
        // ones are the input under test.
        unsafe {
            let n = core::ptr::null_mut::<f64>();
            assert_eq!(
                sl_bistatic_range(&raw const sat, &raw const tx, 0.02, &raw const rx, n),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_bistatic_range_rate(&raw const sat, &raw const tx, 0.02, &raw const rx, n),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_tdoa(&raw const sat, 0.0, &raw const tx, &raw const rx, n),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_fdoa(&raw const sat, 0.0, &raw const tx, &raw const rx, 8.4e9, n),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_delta_dor(
                    &raw const sat,
                    0.0,
                    &raw const tx,
                    &raw const rx,
                    1.2,
                    -0.4,
                    n
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_dsn_sequential_range(
                    segments.as_ptr(),
                    segments.len(),
                    BAND_X,
                    100.0,
                    101.0,
                    26,
                    n
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_dsn_averaged_doppler(
                    segments.as_ptr(),
                    segments.len(),
                    &raw const link,
                    100.0,
                    160.0,
                    n
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_dsn_measured_frequency(
                    segments.as_ptr(),
                    segments.len(),
                    BAND_X,
                    BAND_X,
                    100.0,
                    1.0,
                    n
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_solar_corona_delay(
                    core::ptr::null(),
                    core::ptr::null(),
                    8.4e9,
                    true,
                    core::ptr::null(),
                    0,
                    n
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_orbital_lifetime(
                    &raw const params,
                    initial.as_ptr(),
                    zonals.as_ptr(),
                    zonals.len(),
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_orbital_lifetime_ensemble(
                    &raw const params,
                    initial.as_ptr(),
                    zonals.as_ptr(),
                    zonals.len(),
                    core::ptr::null(),
                    0,
                    core::ptr::null(),
                    0,
                    n,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_ballistic_log_sensitivity(
                    &raw const params,
                    initial.as_ptr(),
                    zonals.as_ptr(),
                    zonals.len(),
                    0.05,
                    n
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_tle_lines(
                    core::ptr::null(),
                    25544,
                    initial.as_ptr(),
                    0.0,
                    core::ptr::null(),
                    0,
                    0,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_lane_wavelengths(1_575.42e6, 1_227.60e6, core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_ionosphere_free(2.0, 1.0, 3.0, 4.0, n),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(sl_wide_lane(2.0, 1.0, 3.0, 4.0, n), SL_ERR_INVALID_ARGUMENT);
            assert_eq!(
                sl_narrow_lane(2.0, 1.0, 3.0, 4.0, n),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_melbourne_wubbena(2.0, 1.0, 3.0, 4.0, 5.0, 6.0, false, n),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_detect_cycle_slips_geometry_free(
                    times.as_ptr(),
                    values.as_ptr(),
                    times.len(),
                    0.1,
                    120.0,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_detect_cycle_slips_polynomial(
                    times.as_ptr(),
                    values.as_ptr(),
                    times.len(),
                    1,
                    3,
                    0.1,
                    120.0,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_ambiguity_decorrelation(
                    float.as_ptr(),
                    float.len(),
                    covariance.as_ptr(),
                    n,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_ambiguity_bootstrap(
                    float.as_ptr(),
                    float.len(),
                    covariance.as_ptr(),
                    core::ptr::null_mut(),
                    n
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_resolve_ambiguities(
                    float.as_ptr(),
                    float.len(),
                    covariance.as_ptr(),
                    2,
                    core::ptr::null_mut(),
                    n,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_collision_probability(&raw const geometry, core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_conjunction_usage_violations(
                    &raw const geometry,
                    5_400.0,
                    Some(rectilinear_sample),
                    core::ptr::null_mut(),
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_closest_approaches(
                    Some(drifting_state),
                    core::ptr::null_mut(),
                    10.0,
                    100.0,
                    1.0,
                    1e-3,
                    core::ptr::null_mut(),
                    0,
                    core::ptr::null_mut()
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            // A null callback is the same class of mistake as a null out-pointer, and the two scans
            // are the only entry points here that can be handed one. Both are given a usable `out`,
            // so it is the callback and not the output that is under test.
            assert_eq!(
                sl_conjunction_usage_violations(
                    &raw const geometry,
                    5_400.0,
                    None,
                    core::ptr::null_mut(),
                    &raw mut violations
                ),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                sl_closest_approaches(
                    None,
                    core::ptr::null_mut(),
                    10.0,
                    100.0,
                    1.0,
                    1e-3,
                    core::ptr::null_mut(),
                    0,
                    &raw mut length
                ),
                SL_ERR_INVALID_ARGUMENT
            );
        }
        // `sl_decay_params_init` is total and takes a null as a request to do nothing: it is the
        // one entry point here with no status to report a refusal through.
        // SAFETY: a null pointer is exactly the input it is documented to accept.
        unsafe { sl_decay_params_init(core::ptr::null_mut()) };
    }

    // ── Closed-loop control ───────────────────────────────────────────────────────────────────

    // The control surface is exercised from C and from C++ by `tests/c/smoke.c`,
    // `tests/cpp/smoke.cpp` and `tests/cpp/nothrow.cpp`, and those are the tests a consumer's
    // compiler runs. They are driven by ctest under `just capi`, which `cargo mutants` does not
    // invoke -- so the first block over this section measured 101 mutants and caught **none**.
    // Nothing below duplicates a C assertion for its own sake; each one names a value the flat
    // ABI shape can get wrong on the way in or out, where the mutation operator says the suite
    // could not tell.

    /// Row-major identity, the reference attitude everything here is measured against.
    const EYE3: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    /// A body at rest.
    fn at_rest() -> SlVec3 {
        SlVec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// A rotation of `angle` about x, row-major, so a test can say what it turned by.
    fn turn_x(angle: f64) -> [f64; 9] {
        let (s, c) = angle.sin_cos();
        [1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c]
    }

    /// A deliberately asymmetric inertia: every principal moment distinct, so a transposed or
    /// re-indexed read of the nine `double` changes the answer instead of cancelling.
    const INERTIA: [f64; 9] = [4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0];

    /// The gain a caller who has read the header would pass.
    const KP: f64 = 8.0;
    /// See [`KP`].
    const KD: f64 = 12.0;

    /// `to_rows` and `from_rows` are the only places the flat ABI shape is interpreted, and both
    /// were mutable and uncaught: a `[[0.0; N]; N]`, a transposed index, an `i * COLS + j` turned
    /// into `i + COLS * j`. The one input that separates all of them is a matrix that is neither
    /// symmetric nor constant, read back through an entry point that returns it re-flattened.
    ///
    /// `sl_clohessy_wiltshire_plant` is that entry point: `A` for the CW plant at mean motion `n`
    /// is a fixed 6x6 whose `(3,4) = 2n` and `(4,3) = -2n` differ in sign, so a transposition is
    /// visible, and whose `(3,0) = 3n^2` is the only entry of its magnitude, so a re-index is too.
    #[test]
    fn the_flat_matrix_shape_survives_the_round_trip() {
        let n = 1.1e-3_f64;
        let (mut a, mut b) = ([0.0_f64; 36], [0.0_f64; 18]);
        // SAFETY: both buffers are the sizes the header documents.
        assert_eq!(
            unsafe { sl_clohessy_wiltshire_plant(n, a.as_mut_ptr(), b.as_mut_ptr()) },
            SL_OK
        );
        // Hill's equations, written out: x'' = 3n^2 x + 2n y', y'' = -2n x', z'' = -n^2 z.
        let at = |i: usize, j: usize| a[i * 6 + j];
        assert!(
            (at(3, 0) - 3.0 * n * n).abs() < 1e-18,
            "A[3][0] is {}",
            at(3, 0)
        );
        assert!(
            (at(3, 4) - 2.0 * n).abs() < 1e-15,
            "A[3][4] is {}",
            at(3, 4)
        );
        assert!(
            (at(4, 3) + 2.0 * n).abs() < 1e-15,
            "A[4][3] is {}",
            at(4, 3)
        );
        assert!((at(5, 2) + n * n).abs() < 1e-18, "A[5][2] is {}", at(5, 2));
        // The three velocity rows of `A` are the identity block, and every other entry is zero:
        // a constant-fill mutant fails on the first of those and a re-index on the second.
        for i in 0..3 {
            for j in 0..6 {
                let want = f64::from(u8::from(j == i + 3));
                assert!(
                    (at(i, j) - want).abs() < 1e-18,
                    "A[{i}][{j}] is {}",
                    at(i, j)
                );
            }
        }
        // `B` selects the three accelerations, so it is the 3x3 identity under a 3x3 of zeros --
        // the sparsest possible witness that the (6,3) flattening is row-major and not column.
        for i in 0..6 {
            for j in 0..3 {
                let want = f64::from(u8::from(i == j + 3));
                let got = b[i * 3 + j];
                assert!((got - want).abs() < 1e-18, "B[{i}][{j}] is {got}");
            }
        }
    }

    /// `from_rows` re-flattens a 3x3 on the way out of the loop, and `i / 3` / `i % 3` were both
    /// mutable and uncaught. A run of one step whose commanded torque is zero leaves the initial
    /// attitude in place, and the initial attitude here is a rotation about x by an angle that is
    /// not a multiple of pi/2 -- so every one of the nine entries is distinct in magnitude or sign
    /// and any re-indexing of the flattening shows up.
    #[test]
    fn the_loop_returns_the_attitude_it_was_given_when_it_has_nothing_to_correct() {
        let start = turn_x(0.37);
        let state = SlAttitudeState {
            rotation: start,
            rate: at_rest(),
        };
        let latency = SlLatency {
            sensor_s: 0.0,
            compute_s: 0.0,
            actuator_s: 0.0,
        };
        let mut end = SlAttitudeState {
            rotation: [0.0; 9],
            rate: at_rest(),
        };
        let mut samples = [SlLoopSample {
            time_s: 0.0,
            error_angle_rad: 0.0,
            torque: at_rest(),
        }; 4];
        let mut len = 0_usize;
        // The reference *is* the attitude the loop starts at, so there is nothing to correct: the
        // body coasts at zero rate and the attitude it ends at is the one it began with, entry for
        // entry against the same rotation. That cannot pass on a plausible-looking matrix.
        // SAFETY: every pointer is to a live local of the documented size.
        let status = unsafe {
            sl_attitude_loop_run(
                LAW_QUATERNION_PD,
                KP,
                KD,
                &raw const latency,
                0.5,
                &raw const state,
                &raw const state,
                start.as_ptr(),
                INERTIA.as_ptr(),
                2,
                samples.as_mut_ptr(),
                samples.len(),
                &raw mut len,
                &raw mut end,
            )
        };
        assert_eq!(status, SL_OK);
        assert_eq!(len, 2, "two steps were asked for");
        for (i, (&got, &want)) in end.rotation.iter().zip(start.iter()).enumerate() {
            assert!((got - want).abs() < 1e-9, "entry {i}: {got} vs {want}");
        }
        // The error is zero at every sample because the reference *is* the attitude.
        for s in &samples[..len] {
            assert!(
                s.error_angle_rad.abs() < 1e-9,
                "error {}",
                s.error_angle_rad
            );
        }
    }

    /// The two laws differ, and `CapiLaw`'s two `match` arms were the largest single cluster of
    /// survivors after `capi_solve_lqr`: nine on `small_angle_gains` and three on `torque`.
    ///
    /// What separates the arms is the linearisation each law states about itself. Both act on an
    /// attitude *parameter* rather than on the angle, and the two parameters shrink at different
    /// rates near identity -- the quaternion vector part is `theta/2`, the MRP `theta/4` -- so
    /// Wie's law linearises to `(Kp/2, Kd)` and Schaub & Junkins' to `(Kp/4, Kd)`. Asserting both
    /// closed forms pins which arm each tag reaches; a swap of the arms changes the proportional
    /// gain by a factor of two and leaves the derivative gain alone, which is why both are checked.
    #[test]
    fn each_attitude_law_reaches_its_own_arm() {
        for (law, share) in [(LAW_QUATERNION_PD, 0.5), (LAW_MRP_LYAPUNOV, 0.25)] {
            let (mut kp, mut kd) = (0.0_f64, 0.0_f64);
            // SAFETY: both out-parameters are live locals.
            assert_eq!(
                unsafe { sl_attitude_small_angle_gains(law, KP, KD, &raw mut kp, &raw mut kd) },
                SL_OK
            );
            assert!((kp - share * KP).abs() < 1e-12, "law {law}: Kp is {kp}");
            assert!((kd - KD).abs() < 1e-12, "law {law}: Kd is {kd}");
        }

        // The same linearisation, reached through the other entry point.
        // `sl_quaternion_pd_response` takes the law's *own* gains and is fixed to the quaternion
        // law, so `omega_n = sqrt(Kp' / J)` and `zeta = Kd' / (2 sqrt(Kp' J))` must be built on
        // the `Kp' = Kp/2` its sibling reports -- which is what ties the two entry points to one
        // linearisation instead of two that happen to agree.
        let (mut wn, mut zeta) = (0.0_f64, 0.0_f64);
        // SAFETY: both out-parameters are live locals.
        assert_eq!(
            unsafe { sl_quaternion_pd_response(KP, KD, 4.0, &raw mut wn, &raw mut zeta) },
            SL_OK
        );
        let want_wn = (0.5 * KP / 4.0).sqrt();
        assert!((wn - want_wn).abs() < 1e-12, "omega_n is {wn} vs {want_wn}");
        let want_zeta = KD / (2.0 * (4.0 * 0.5 * KP).sqrt());
        assert!(
            (zeta - want_zeta).abs() < 1e-12,
            "zeta is {zeta} vs {want_zeta}"
        );
    }

    /// `ControlLaw::torque`'s two arms, pinned by direction rather than by magnitude.
    ///
    /// A body turned about `+x` from its reference and at rest must be commanded a moment about
    /// `-x` and about nothing else -- that is what "restoring" means, and it is true of both laws
    /// with no shared constant between them. A whole-function replacement by `[0.0; 3]` or
    /// `[1.0; 3]` fails the sign; a swap of the two arms does not, which is why the magnitudes
    /// are compared against each other as well. At the same gains and the same small angle the
    /// quaternion law's moment is the larger, because the quaternion vector part is twice the MRP
    /// -- the same factor of two the gains report, arrived at through the torque instead.
    #[test]
    fn both_laws_command_a_restoring_moment() {
        let turned = turn_x(0.20);
        let rest = at_rest();
        let mut moment = [rest; 2];
        for (i, law) in [LAW_QUATERNION_PD, LAW_MRP_LYAPUNOV]
            .into_iter()
            .enumerate()
        {
            // SAFETY: every pointer is to a live local of the documented size.
            assert_eq!(
                unsafe {
                    sl_attitude_torque(
                        law,
                        KP,
                        KD,
                        turned.as_ptr(),
                        EYE3.as_ptr(),
                        &raw const rest,
                        &raw const rest,
                        INERTIA.as_ptr(),
                        &raw mut moment[i],
                    )
                },
                SL_OK
            );
            assert!(moment[i].x < 0.0, "law {law}: torque.x is {}", moment[i].x);
            assert!(
                moment[i].y.abs() < 1e-12,
                "law {law}: torque.y is {}",
                moment[i].y
            );
            assert!(
                moment[i].z.abs() < 1e-12,
                "law {law}: torque.z is {}",
                moment[i].z
            );
        }
        assert!(
            moment[0].x < moment[1].x,
            "the quaternion law must command the larger moment: {} vs {}",
            moment[0].x,
            moment[1].x
        );
    }

    /// `capi_solve_lqr` is 48 of the 88 survivors, every one of them a whole-function replacement
    /// by some `Ok((vec![c], vec![c], c))`. The test that kills all of them at once is the one
    /// that does not trust the solver: recompute `K = R^-1 B^T P` from the `P` the ABI returned,
    /// in this test, and check the Riccati residual the ABI reports against the equation itself.
    ///
    /// A constant `P` cannot satisfy `A^T P + P A - P B R^-1 B^T P + Q = 0` for this plant, and a
    /// `K` that does not equal `R^-1 B^T P` is not the gain of the `P` beside it, whatever either
    /// of them is. Neither assertion can be satisfied by a plausible-looking pair.
    #[test]
    fn the_lqr_gain_is_the_one_its_own_riccati_solution_implies() {
        // A double integrator: x' = v, v' = u. Two states, one input -- the smallest of the four
        // instantiated shapes, and the only one whose answer is a closed form worth checking.
        let a = [0.0, 1.0, 0.0, 0.0];
        let b = [0.0, 1.0];
        let q = [1.0, 0.0, 0.0, 1.0];
        let r = [1.0];
        let (mut gain, mut riccati, mut residual) = ([0.0_f64; 2], [0.0_f64; 4], 0.0_f64);
        // SAFETY: the buffers are (m*n), (n*n) and one, the sizes the header documents for (2,1).
        assert_eq!(
            unsafe {
                sl_lqr(
                    a.as_ptr(),
                    b.as_ptr(),
                    q.as_ptr(),
                    r.as_ptr(),
                    2,
                    1,
                    gain.as_mut_ptr(),
                    riccati.as_mut_ptr(),
                    &raw mut residual,
                )
            },
            SL_OK
        );
        assert!(
            residual < 1e-9,
            "the solver reports its own residual as {residual}"
        );

        // `K = R^-1 B^T P`, computed here from what came back rather than taken on trust. `B`
        // selects the second state, so `B^T P` is the second row of `P`.
        for j in 0..2 {
            let want = riccati[2 + j] / r[0];
            assert!(
                (gain[j] - want).abs() < 1e-9,
                "K[{j}] is {} vs {want}",
                gain[j]
            );
        }
        // The Riccati residual, recomputed: `A^T P + P A - P B R^-1 B^T P + Q`.
        let p = |i: usize, j: usize| riccati[i * 2 + j];
        let am = |i: usize, j: usize| a[i * 2 + j];
        for i in 0..2 {
            for j in 0..2 {
                let mut e = q[i * 2 + j];
                for k in 0..2 {
                    e += am(k, i) * p(k, j) + p(i, k) * am(k, j);
                }
                e -= p(i, 1) * p(1, j) / r[0];
                assert!(e.abs() < 1e-8, "Riccati residual at ({i},{j}) is {e}");
            }
        }
        // `P` is symmetric positive definite, which no constant fill of both signs can be.
        assert!((p(0, 1) - p(1, 0)).abs() < 1e-12, "P is not symmetric");
        assert!(
            p(0, 0) > 0.0 && p(0, 0) * p(1, 1) > p(0, 1) * p(1, 0),
            "P is not definite"
        );
    }

    /// The three walls this ABI invents because C has no const parameter, each refused with the
    /// status that says *which* wall it is. `check_agents`, `check_matrix_order` and
    /// `check_lqr_shape` were all replaceable by `Ok(())` with nothing noticing.
    #[test]
    fn the_monomorphisation_walls_refuse_from_both_sides() {
        // Agents: 2..=12 for the consensus family. One below and one above, and the two ends
        // admitted -- a `!` deleted from the range test inverts exactly this pair.
        let two = [0.0, 1.0, 1.0, 0.0];
        let mut connected = false;
        for (agents, want) in [
            (1_usize, SL_ERR_INVALID_ARGUMENT),
            (13, SL_ERR_INVALID_ARGUMENT),
        ] {
            // SAFETY: `two` is longer than the 1-agent case reads and the 13-agent case is
            // refused before the pointer is read at all.
            let status = unsafe { sl_graph_is_connected(two.as_ptr(), agents, &raw mut connected) };
            assert_eq!(status, want, "{agents} agents");
        }
        // SAFETY: four `double` is exactly the 2-agent adjacency.
        assert_eq!(
            unsafe { sl_graph_is_connected(two.as_ptr(), 2, &raw mut connected) },
            SL_OK
        );
        assert!(connected, "a single edge joins two agents");

        // Matrix order: the same two numbers, a different noun. A companion matrix of order 2.
        let companion = [0.0, -2.0, 1.0, -3.0];
        let mut coefficients = [0.0_f64; 2];
        for order in [1_usize, 13] {
            // SAFETY: as above -- either the buffer is long enough or the call refuses first.
            assert_eq!(
                unsafe {
                    sl_characteristic_polynomial(
                        companion.as_ptr(),
                        order,
                        coefficients.as_mut_ptr(),
                    )
                },
                SL_ERR_INVALID_ARGUMENT,
                "order {order}"
            );
        }
        // Cayley-Hamilton on a companion matrix: the polynomial *is* the last column, negated.
        // SAFETY: both buffers are the order-2 sizes.
        assert_eq!(
            unsafe {
                sl_characteristic_polynomial(companion.as_ptr(), 2, coefficients.as_mut_ptr())
            },
            SL_OK
        );
        assert!(
            (coefficients[0] - 2.0).abs() < 1e-12,
            "c0 is {}",
            coefficients[0]
        );
        assert!(
            (coefficients[1] - 3.0).abs() < 1e-12,
            "c1 is {}",
            coefficients[1]
        );

        // LQR shape: four instantiated `(n, m)`, and (2,2) is not one of them even though both
        // numbers are individually admissible -- which is the whole reason it is its own check.
        let (a, b, q, r) = ([0.0_f64; 4], [0.0_f64; 4], [0.0_f64; 4], [0.0_f64; 4]);
        let (mut gain, mut riccati, mut residual) = ([0.0_f64; 4], [0.0_f64; 4], 0.0_f64);
        // SAFETY: every buffer is at least the (2,2) size the call is refused for.
        assert_eq!(
            unsafe {
                sl_lqr(
                    a.as_ptr(),
                    b.as_ptr(),
                    q.as_ptr(),
                    r.as_ptr(),
                    2,
                    2,
                    gain.as_mut_ptr(),
                    riccati.as_mut_ptr(),
                    &raw mut residual,
                )
            },
            SL_ERR_INVALID_ARGUMENT
        );
    }

    /// `check_after` is eight survivors on its own, and the reason it needs its own test is that
    /// its two refusals carry *different* statuses. A run to `t1 <= 0` is a malformed argument;
    /// a run to a positive `t1` that is shorter than its own delay is `SL_ERR_DOMAIN`, because
    /// every number in it is individually admissible and only their relation is not.
    ///
    /// The three points either side of the boundary are what the mutation operators move: `>`
    /// into `>=` or `<`, `<=` into `>`, `&&` into `||`, and the whole predicate into `Ok(())`.
    #[test]
    fn a_run_shorter_than_its_own_delay_is_a_domain_error() {
        let two = [0.0, 1.0, 1.0, 0.0];
        let initial = [1.0, -1.0];
        let mut final_state = [0.0_f64; 2];
        let (mut disagreement, mut accepted, mut rejected) = (0.0_f64, 0_u32, 0_u32);
        let mut run = |t1: f64, delay: f64| {
            // SAFETY: `two` and `initial` are the 2-agent sizes and the outputs are live locals.
            unsafe {
                sl_consensus_run(
                    two.as_ptr(),
                    2,
                    CONSENSUS_DELAYED_RELATIVE,
                    delay,
                    initial.as_ptr(),
                    t1,
                    1e-9,
                    final_state.as_mut_ptr(),
                    &raw mut disagreement,
                    &raw mut accepted,
                    &raw mut rejected,
                )
            }
        };
        assert_eq!(run(0.0, 0.5), SL_ERR_INVALID_ARGUMENT, "a run to zero");
        assert_eq!(
            run(-1.0, 0.5),
            SL_ERR_INVALID_ARGUMENT,
            "a run to a negative time"
        );
        assert_eq!(run(f64::NAN, 0.5), SL_ERR_INVALID_ARGUMENT, "a run to NaN");
        assert_eq!(run(0.5, 0.5), SL_ERR_DOMAIN, "a run exactly one delay long");
        assert_eq!(
            run(0.25, 0.5),
            SL_ERR_DOMAIN,
            "a run shorter than its delay"
        );
        assert_eq!(run(20.0, 0.5), SL_OK, "a run long enough to take an arc");
        // Consensus over a connected graph drives the disagreement down, and the two agents
        // started one apart -- so this also says `run_consensus` integrated rather than returned
        // its input. The `0.05 * delay` first step is the other mutated product in that function.
        assert!(
            disagreement < 1e-3,
            "the disagreement is still {disagreement}"
        );
        assert!(accepted > 0, "no step was accepted");
        // The first trial step this ABI hands the adaptive driver is `0.05 * delay`, and
        // `out_rejected` is where that number is observable. A seed a twentieth of the link delay
        // is small enough that the DOP853 controller never has to reject one; measured on this
        // exact run, `0.05 + delay` costs 13 rejections and `0.05 / delay` costs 3, while all
        // three agree on the final state to 1e-9. The seed is invisible in the answer and visible
        // in the work, which is the only place an assertion can reach it.
        assert_eq!(rejected, 0, "the step seed cost {rejected} rejected steps");
        assert!(
            (final_state[0] - final_state[1]).abs() < 1e-3,
            "the states are {} and {}",
            final_state[0],
            final_state[1]
        );
    }
}
