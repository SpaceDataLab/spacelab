//! The PyO3 bridge of the spacelab core: Python bindings for the `spacelab-*` crates.
//!
//! [`register`] adds the bindings to a Python module, which belongs to the cdylib that links this
//! crate: the `spacedatalab` wheel on PyPI, built by
//! [SpaceDataLab/spacedatalab](https://github.com/SpaceDataLab/spacedatalab), or
//! `spacelab-python-harness`, which loads the bridge for spacelab's own Python tests.
//!
//! ```python
//! import spacedatalab as sl
//! rf, vf = sl.kepler_propagate([7000e3, 0, 0], [0, 7.546e3, 0], 3.986004418e14, 600.0)
//! a = sl.geopotential_acceleration(70, [6778e3, 0, 0])   # XGM2019e to degree 70
//! ```
//!
//! Vectors cross the boundary as lists or tuples; an input a kernel rejects raises `ValueError`.

// The shipped artefact must never be built without `extension-module`. Without it PyO3 links
// libpython directly, which imports fine on the machine that built it and then breaks the
// `abi3-py311` promise everywhere else — a wheel that loads under one interpreter and crashes under
// the next. Testing is the one context where linking libpython is the point, so the invariant is
// expressible rather than merely documented: a bare `--no-default-features` is a compile error, and
// the way out of it is to say `test-interpreter`, which no wheel build has a reason to name.
#[cfg(all(
    not(feature = "extension-module"),
    not(feature = "test-interpreter"),
    not(test)
))]
compile_error!(
    "spacelab-python is linked into a CPython extension and needs the `extension-module` feature; \
     `--no-default-features` is for `cargo test --features test-interpreter` only"
);

// `auto-initialize` starts an interpreter for the test process, which is what lets a `#[test]` call
// an entry point taking `Python<'_>`. It has to link libpython to do that, so it cannot coexist with
// `extension-module`, and cargo features are additive — nothing stops someone asking for both.
#[cfg(all(feature = "extension-module", feature = "test-interpreter"))]
compile_error!(
    "`test-interpreter` links libpython and `extension-module` promises not to; \
     use `--no-default-features --features test-interpreter`"
);

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use spacelab_astro::{Cr3bp, lambert as astro_lambert, propagate as kepler, richardson_halo_guess};
use spacelab_astrometry::reduce::Aberration;
use spacelab_autodiff::{DoubleDouble, Scalar};
use spacelab_constellation::{
    DraimTetrahedral, FlowerConstellation, LatticeFlower, StreetOfCoverage, Walker, WalkerPattern,
};
use spacelab_control::attitude::{AttitudeError, ControlLaw, MrpLyapunov, QuaternionPd};
use spacelab_control::closed_loop::{AttitudeLoop, AttitudeState};
use spacelab_control::delayed::{InitialValue, breakpoints, integrate_delayed};
use spacelab_control::formation::{ConsensusNetwork, Graph, GraphError, Protocol};
use spacelab_control::latency::Latency;
use spacelab_control::margin::{
    SecondOrderLoop, SmallAngleGains, worst_principal_axis as control_worst_axis,
};
use spacelab_control::orbit::{
    ElementFeedback, characteristic_polynomial as control_characteristic_polynomial,
    clohessy_wiltshire_plant as control_cw_plant, lqr as control_lqr,
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
use spacelab_earthvision::camera::{
    PinholeCamera as EvPinholeCamera, triangulate as ev_triangulate,
};
use spacelab_earthvision::features::{
    Corner, HarrisConfig, Image, detect_corners as ev_detect_corners,
};
use spacelab_earthvision::mvs::plane_sweep_mvs as ev_plane_sweep;
use spacelab_earthvision::pansharpen::{
    brovey_pan_sharpen as ev_brovey, gram_schmidt_pan_sharpen as ev_gram_schmidt,
};
use spacelab_earthvision::rpc::RationalPolynomialCamera;
use spacelab_earthvision::rpc_adjust::{
    AdjustmentModel, GroundControlPoint, RpcAdjustment,
    estimate_adjustment as ev_estimate_adjustment, forward_adjusted as ev_forward_adjusted,
};
use spacelab_earthvision::sfm::{Observation2d, Track, build_tracks as ev_build_tracks};
use spacelab_earthvision::stereo::{SgmConfig, semi_global_match as ev_sgm};
use spacelab_earthvision::swath::{
    SwathGeometry, ground_point_at as ev_ground_point_at,
    horizon_look_angle as ev_horizon_look_angle, swath_geometry as ev_swath_geometry,
};
use spacelab_elements::design::{
    SUN_SYNCHRONOUS_NODE_RATE, critical_inclination as design_critical_inclination,
    frozen_eccentricity as design_frozen_eccentricity,
    repeat_ground_track_sma as design_repeat_ground_track_sma,
    sun_synchronous_inclination as design_sun_synchronous_inclination,
};
use spacelab_elements::{BrouwerLyddane, KeplerianElements};
use spacelab_ephemeris::{Ephemeris as De440Ephemeris, Epoch as EphEpoch, Frame as EphFrame};
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
    CioInterpolator, EARTH_ROTATION_RATE, Gcrf, Position, StateVector, Teme,
    gcrf_to_itrf as frames_gcrf_to_itrf, teme_to_itrf as frames_teme_to_itrf,
};
use spacelab_geomagnetism::igrf::field_ecef as gnss_igrf_field;
use spacelab_gnss::{
    FrequencyPair, GeometryFreeDetector, GnssSystem, ObsHeader, PolynomialFitDetector, Reduction,
    RinexObs, SlipVerdict, TimeSystem, VmfGrid, carrier_frequency as gnss_carrier_frequency,
    dop as gnss_dop, geometry_free as gnss_geometry_free, glonass_l1_frequency as gnss_glonass_l1,
    glonass_l2_frequency as gnss_glonass_l2, gmf_mapping as gnss_gmf_mapping, gpt3 as gnss_gpt3,
    graphic as gnss_graphic, higher_order_iono::second_order_group_delay_geodetic as gnss_hoi2,
    klobuchar_iono_delay as gnss_klobuchar, niell_mapping as gnss_niell_mapping, parse_glonass_nav,
    parse_nav, saastamoinen_zenith_hydrostatic as gnss_saastamoinen,
    vmf1_mapping as gnss_vmf1_mapping, vmf3_mapping as gnss_vmf3_mapping,
};
use spacelab_ionosphere::{
    DateTimeComponents, GeodeticPoint, NeQuickGalileo, path_delay as iono_path_delay,
};
use spacelab_propagation::attitude::Inertia;
use spacelab_propagation::density_calibration::{DensityCorrection, DensitySample};
use spacelab_propagation::density_data::{grid_samples, orbit_mean_samples, parse_density};
use spacelab_propagation::density_rom::{DensityRom, DmdcModel, ReducedDynamics};
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
    DecayAtmosphere, DecayModel, Lifetime,
    ballistic_log_sensitivity as estimation_ballistic_sensitivity,
    ensemble_lifetime as estimation_ensemble_lifetime, lifetime as estimation_lifetime,
};
use spacelab_propagation::relativity::EARTH_ANGULAR_MOMENTUM_PER_MASS;
use spacelab_propagation::sdtm::{
    Envelope, SdtmAtmosphere, SdtmDrivers, SdtmPoint,
    coefficients::{
        ENVELOPE as SDTM1_ENVELOPE, FIELD as SDTM1_FIELD, UNCERTAINTY as SDTM1_UNCERTAINTY,
    },
};
use spacelab_propagation::space_weather::{
    density_drivers, nrlmsise_ap_array, parse_space_weather,
};
use spacelab_propagation::symplectic::yoshida8_scalar;
use spacelab_propagation::time_variable_gravity::{TimeVariableField, parse_icgem_time_variable};
use spacelab_propagation::{
    DeSitter, Dop853, DsstMeanPropagator, DsstSolarRadiationPressure, DsstTesseral, DsstThirdBody,
    DsstZonal, GeoPotential, LenseThirring, Nrlmsise00Atmosphere, Nrlmsise00Flags, Nrlmsise00Input,
    Schwarzschild, SolidEarthTide, Tolerance, conventional_zonal_coefficients as tvg_conventional,
    gtd7d, secular_zonal_correction as tvg_secular, sentman_panel,
    speed_ratio as sentman_speed_ratio,
};
use spacelab_sensors::{AgileOptical, ElevationMask, OrbitBound, SideLookingSAR};
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

const MU_EARTH: f64 = 3.986_004_418e14;
const RE_EARTH: f64 = 6_378_137.0;

fn vec3(a: [f64; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

fn arr(v: Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

/// Linear interpolation of a sorted `(ts, ys)` series at `t`, clamped at the ends — for feeding a
/// time-varying EOP series into a propagation frame.
fn lerp_at(ts: &[f64], ys: &[f64], t: f64) -> f64 {
    let n = ts.len();
    if t <= ts[0] {
        return ys[0];
    }
    if t >= ts[n - 1] {
        return ys[n - 1];
    }
    let i = ts.partition_point(|&x| x <= t) - 1;
    let frac = (t - ts[i]) / (ts[i + 1] - ts[i]);
    ys[i] + frac * (ys[i + 1] - ys[i])
}

/// Two-body (Kepler) propagation: advance state `(r, v)` by `dt` under gravity `mu`. Consistent SI-ish
/// units (position, position/time, length³/time²). Returns `(r, v)` at `t + dt`.
#[pyfunction]
fn kepler_propagate(r: [f64; 3], v: [f64; 3], mu: f64, dt: f64) -> ([f64; 3], [f64; 3]) {
    let (rf, vf) = kepler(vec3(r), vec3(v), mu, dt);
    (arr(rf), arr(vf))
}

/// Lambert transfer: the velocities `(v1, v2)` connecting positions `r1` and `r2` in time `tof` under
/// `mu`. `prograde` selects the short-way (prograde) branch. Raises `ValueError` if no arc exists.
#[pyfunction]
#[pyo3(signature = (r1, r2, tof, mu, prograde=true))]
fn lambert(
    py: Python<'_>,
    r1: [f64; 3],
    r2: [f64; 3],
    tof: f64,
    mu: f64,
    prograde: bool,
) -> PyResult<([f64; 3], [f64; 3])> {
    // Released off the GIL so a Python-threaded porkchop grid (many independent solves)
    // runs its Lambert legs in parallel rather than serialized on the interpreter lock.
    py.detach(|| {
        astro_lambert(vec3(r1), vec3(r2), tof, mu, prograde)
            .map(|(a, b)| (arr(a), arr(b)))
            .ok_or_else(|| PyValueError::new_err("no Lambert solution for the given transfer"))
    })
}

/// Richardson's third-order analytic halo seed: the perpendicular-crossing synodic state
/// `[x, 0, z, 0, vy, 0]` for collinear point `li` (0 = L1, 1 = L2), γ-normalized out-of-plane amplitude
/// `az`, class `northern`, and mass ratio `mu` (default Earth–Moon). Raises `ValueError` if invalid.
#[pyfunction]
#[pyo3(signature = (li, az, northern=true, mu=0.012_150_584_460_351))]
fn richardson_halo(li: usize, az: f64, northern: bool, mu: f64) -> PyResult<[f64; 6]> {
    richardson_halo_guess(Cr3bp { mu }, li, az, northern)
        .ok_or_else(|| PyValueError::new_err("invalid libration point index or amplitude"))
}

/// Tesseral geopotential acceleration from the embedded model (XGM2019e) truncated to `degree`
/// (≤ 70), at body-fixed position `r` (metres). Returns `[ax, ay, az]` in m/s². Raises `ValueError`
/// if `degree` exceeds the embedded table.
#[pyfunction]
fn geopotential_acceleration(degree: usize, r: [f64; 3]) -> PyResult<[f64; 3]> {
    let g = GeoPotential::embedded(degree)
        .ok_or_else(|| PyValueError::new_err("degree exceeds the embedded model (max 70)"))?;
    Ok(g.acceleration(0.0, &r, &[0.0; 3]))
}

/// The five libration points of the circular restricted three-body problem for mass ratio `mu`, each
/// as `[x, y, z]` in synodic nondimensional coordinates: `[L1, L2, L3, L4, L5]`.
#[pyfunction]
#[pyo3(signature = (mu=0.012_150_584_460_351))]
fn libration_points(mu: f64) -> [[f64; 3]; 5] {
    Cr3bp { mu }.libration_points().map(arr)
}

/// SGP4/SDP4 propagation of a two-line element set to a signed offset `minutes` from its epoch.
/// Returns `(r, v)` in the TEME frame (metres, metres/second). Raises `ValueError` on a malformed TLE
/// or a propagation failure (decayed orbit, etc.).
#[pyfunction]
fn sgp4_propagate(line1: &str, line2: &str, minutes: f64) -> PyResult<([f64; 3], [f64; 3])> {
    let sat = Satellite::from_tle(line1, line2)
        .map_err(|e| PyValueError::new_err(format!("invalid TLE: {e:?}")))?;
    let sv = sat
        .propagate_minutes(minutes)
        .map_err(|e| PyValueError::new_err(format!("SGP4 propagation failed: {e:?}")))?;
    Ok((arr(sv.position.vector), arr(sv.velocity.vector)))
}

/// A ground-cell coverage constraint (sensor), built with one of the static constructors and passed to
/// [`coverage_metrics`]. Angles are degrees.
// pyo3 0.29 makes the `FromPyObject` derive for `Clone` pyclasses opt-in; keep the prior behaviour.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Sensor {
    kind: SensorKind,
}

#[derive(Clone)]
enum SensorKind {
    Elevation(ElevationMask),
    Sar(SideLookingSAR),
    Agile(AgileOptical),
}

#[pymethods]
impl Sensor {
    /// Ground-station / nadir elevation mask: covered when the satellite rises above `min_elevation_deg`.
    #[staticmethod]
    fn elevation(min_elevation_deg: f64) -> PyResult<Self> {
        ElevationMask::new(min_elevation_deg.to_radians())
            .map(|e| Self {
                kind: SensorKind::Elevation(e),
            })
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))
    }

    /// The same mask stated as an **apparent** elevation — what a station's "visible above 10°"
    /// actually means — with the atmosphere converted into the threshold once.
    ///
    /// [`Sensor.elevation`] takes the *geometric* elevation, the straight line of sight before the
    /// atmosphere bends it. The two differ by 5.3′ at 10° and 34′ at the horizon, worth 0.81 s to
    /// 20.3 s of pass edge with the same sign on every edge of every pass, so it does not average
    /// out. The conversion is exact rather than an approximation: `h ↦ h + R(h)` is strictly
    /// increasing, so one apparent mask is exactly one geometric mask, and the scan never sees the
    /// atmosphere.
    #[staticmethod]
    #[pyo3(signature = (min_elevation_deg, pressure_hpa = 1010.0, temperature_k = 283.0))]
    fn elevation_apparent(
        min_elevation_deg: f64,
        pressure_hpa: f64,
        temperature_k: f64,
    ) -> PyResult<Self> {
        Refraction {
            pressure_hpa,
            temperature_k,
        }
        .elevation_mask(ApparentAltitude(min_elevation_deg.to_radians()))
        .map(|e| Self {
            kind: SensorKind::Elevation(e),
        })
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "{min_elevation_deg}° is below the apparent horizon or beyond what the refraction \
                 fit reaches, or the geometric mask it converts to is not one an elevation mask \
                 accepts"
            ))
        })
    }

    /// Side-looking SAR incidence band `[near_deg, far_deg]` (a ring, not a disc).
    #[staticmethod]
    fn sar(near_deg: f64, far_deg: f64) -> PyResult<Self> {
        SideLookingSAR::new(near_deg.to_radians(), far_deg.to_radians())
            .map(|s| Self {
                kind: SensorKind::Sar(s),
            })
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))
    }

    /// Agile optical instrument, slewable up to `max_off_nadir_deg` from nadir.
    #[staticmethod]
    fn agile(max_off_nadir_deg: f64) -> PyResult<Self> {
        AgileOptical::new(max_off_nadir_deg.to_radians())
            .map(|a| Self {
                kind: SensorKind::Agile(a),
            })
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))
    }
}

/// Build ground cells `[(centre, zenith), …]` on a spherical Earth from latitude/longitude grids
/// (degrees). Feed the result to [`coverage_metrics`]. Centres are ECEF metres; zeniths are unit.
#[pyfunction]
fn grid_cells(lats_deg: Vec<f64>, lons_deg: Vec<f64>) -> Vec<([f64; 3], [f64; 3])> {
    let mut out = Vec::with_capacity(lats_deg.len() * lons_deg.len());
    for &lat in &lats_deg {
        for &lon in &lons_deg {
            let (phi, lam) = (lat.to_radians(), lon.to_radians());
            let up = Vec3::new(phi.cos() * lam.cos(), phi.cos() * lam.sin(), phi.sin());
            out.push((arr(up.scale(RE_EARTH)), arr(up)));
        }
    }
    out
}

/// Per-cell coverage/revisit metrics for a circular orbit `(a, inc, raan, phase)` (metres, radians)
/// seen by `sensor`, over `[0, horizon]` seconds, using the production access-window engine
/// (`revisit_map`): analytic ECI→ECEF propagation feeds the same event detector the Rust pipeline uses.
/// Returns, aligned with `cells`, `[passes, coverage_fraction, max_gap_s, mean_gap_s,
/// mean_response_time_s]` per cell — the Phase-6 features, computed from Python without a CSV
/// round-trip. Feed the whole list to [`constellation_fom`] for the aggregate figures of merit.
#[pyfunction]
#[pyo3(signature = (a, inc, raan, phase, sensor, cells, horizon, tol=1.0, threads=8))]
#[allow(clippy::too_many_arguments)]
fn coverage_metrics(
    a: f64,
    inc: f64,
    raan: f64,
    phase: f64,
    sensor: &Sensor,
    cells: Vec<([f64; 3], [f64; 3])>,
    horizon: f64,
    tol: f64,
    threads: usize,
) -> Vec<[f64; 5]> {
    let cells3: Vec<(Vec3, Vec3)> = cells.iter().map(|&(c, z)| (vec3(c), vec3(z))).collect();
    let n = (MU_EARTH / (a * a * a)).sqrt();
    let sat_at = move |t: f64| {
        let u = phase + n * t;
        let (cu, su) = (u.cos(), u.sin());
        let (cr, sr) = (raan.cos(), raan.sin());
        let (ci, si) = (inc.cos(), inc.sin());
        let eci = Vec3::new(
            a * (cu * cr - su * sr * ci),
            a * (cu * sr + su * cr * ci),
            a * (su * si),
        );
        Mat3::rot3(EARTH_ROTATION_RATE * t) * eci
    };
    let orbit = OrbitBound {
        semi_major_axis: a,
        eccentricity: 0.0,
        mu: MU_EARTH,
    };
    let m = match &sensor.kind {
        SensorKind::Elevation(s) => revisit_map(
            &cells3,
            s,
            &orbit,
            EARTH_ROTATION_RATE,
            sat_at,
            0.0,
            horizon,
            tol,
            threads,
        ),
        SensorKind::Sar(s) => revisit_map(
            &cells3,
            s,
            &orbit,
            EARTH_ROTATION_RATE,
            sat_at,
            0.0,
            horizon,
            tol,
            threads,
        ),
        SensorKind::Agile(s) => revisit_map(
            &cells3,
            s,
            &orbit,
            EARTH_ROTATION_RATE,
            sat_at,
            0.0,
            horizon,
            tol,
            threads,
        ),
    };
    m.iter()
        .map(|c| {
            [
                f64::from(c.passes),
                c.coverage_fraction,
                c.max_gap,
                c.mean_gap,
                c.mean_response_time,
            ]
        })
        .collect()
}

/// Aggregate per-cell metrics (each `[passes, coverage_fraction, max_gap, mean_gap,
/// mean_response_time]`, as [`coverage_metrics`] returns) into whole-constellation figures of merit:
/// `(point_count, mean_coverage, min_coverage, worst_max_gap, mean_max_gap, mean_response_time,
/// worst_response_time, fully_covered_points)`. `weights` (e.g. `cos(latitude)`) area-weight the means;
/// `None` weights every point equally.
#[pyfunction]
#[pyo3(signature = (points, weights=None))]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn constellation_fom(
    points: Vec<[f64; 5]>,
    weights: Option<Vec<f64>>,
) -> (u32, f64, f64, f64, f64, f64, f64, u32) {
    let metrics: Vec<CoverageMetrics> = points
        .iter()
        .map(|r| CoverageMetrics {
            passes: r[0] as u32,
            covered_seconds: 0.0,
            coverage_fraction: r[1],
            max_gap: r[2],
            mean_gap: r[3],
            mean_response_time: r[4],
            longest_pass: 0.0,
            shortest_pass: 0.0,
        })
        .collect();
    let f = coverage_constellation_fom(&metrics, weights.as_deref());
    (
        f.point_count,
        f.mean_coverage,
        f.min_coverage,
        f.worst_max_gap,
        f.mean_max_gap,
        f.mean_response_time,
        f.worst_response_time,
        f.fully_covered_points,
    )
}

/// The raw access windows of one circular orbit `(a, inc, raan, phase)` over a single ground `cell`
/// `(centre, zenith)` (from [`grid_cells`]) seen by `sensor`, over `[0, horizon]` seconds. Returns the
/// `(rise, set)` intervals — the engine's primary product, which [`coverage_metrics`] only aggregates.
///
/// This is the rigorous scan, not a fixed-grid sample: the step is bounded by the sensor's `max_rate`,
/// so no pass is stepped over however short, and each edge is refined to `tol` seconds by Brent. A
/// naive cadence-`Δ` sampler instead misplaces every edge by up to `Δ` and drops any pass shorter than
/// `Δ` that falls between samples — the difference `experiments/access_window_precision.py` quantifies.
#[pyfunction]
#[pyo3(signature = (a, inc, raan, phase, sensor, cell, horizon, tol = 1e-3))]
#[allow(clippy::too_many_arguments)]
fn access_windows(
    a: f64,
    inc: f64,
    raan: f64,
    phase: f64,
    sensor: &Sensor,
    cell: ([f64; 3], [f64; 3]),
    horizon: f64,
    tol: f64,
) -> Vec<(f64, f64)> {
    let n = (MU_EARTH / (a * a * a)).sqrt();
    let sat_at = move |t: f64| {
        let u = phase + n * t;
        let (cu, su) = (u.cos(), u.sin());
        let (cr, sr) = (raan.cos(), raan.sin());
        let (ci, si) = (inc.cos(), inc.sin());
        let eci = Vec3::new(
            a * (cu * cr - su * sr * ci),
            a * (cu * sr + su * cr * ci),
            a * (su * si),
        );
        Mat3::rot3(EARTH_ROTATION_RATE * t) * eci
    };
    let orbit = OrbitBound {
        semi_major_axis: a,
        eccentricity: 0.0,
        mu: MU_EARTH,
    };
    let (centre, zenith) = (vec3(cell.0), vec3(cell.1));
    let mut out = Vec::new();
    let mut emit = |w: Window| out.push((w.rise, w.set));
    let (t0, rate) = (0.0, EARTH_ROTATION_RATE);
    match &sensor.kind {
        SensorKind::Elevation(s) => {
            coverage_access_windows(
                s, &orbit, rate, centre, zenith, sat_at, t0, horizon, tol, &mut emit,
            );
        }
        SensorKind::Sar(s) => {
            coverage_access_windows(
                s, &orbit, rate, centre, zenith, sat_at, t0, horizon, tol, &mut emit,
            );
        }
        SensorKind::Agile(s) => {
            coverage_access_windows(
                s, &orbit, rate, centre, zenith, sat_at, t0, horizon, tol, &mut emit,
            );
        }
    }
    out
}

/// TEME → ITRF (Earth-fixed) frame transform of a state `(r, v)` in metres. `ut1_ns` / `tt_ns` are
/// nanoseconds since J2000 in the UT1 and TT scales; `xp_rad`, `yp_rad` are polar-motion angles and
/// `lod_seconds` the excess length of day (all default to zero for a GMST-only rotation). Returns the
/// ITRF `(r, v)`.
#[pyfunction]
#[pyo3(signature = (r_teme, v_teme, ut1_ns, tt_ns, xp_rad=0.0, yp_rad=0.0, lod_seconds=0.0))]
fn teme_to_itrf(
    r_teme: [f64; 3],
    v_teme: [f64; 3],
    ut1_ns: i128,
    tt_ns: i128,
    xp_rad: f64,
    yp_rad: f64,
    lod_seconds: f64,
) -> ([f64; 3], [f64; 3]) {
    let ut1 = Epoch::<Ut1>::from_ns_since_j2000(ut1_ns);
    let tt = Epoch::<Tt>::from_ns_since_j2000(tt_ns);
    let tf = frames_teme_to_itrf(
        ut1,
        tt,
        Angle::from_radians(xp_rad),
        Angle::from_radians(yp_rad),
        lod_seconds,
    );
    let sv = tf.state(StateVector::<Teme>::new(vec3(r_teme), vec3(v_teme)));
    (arr(sv.position.vector), arr(sv.velocity.vector))
}

/// GCRF (GCRS) state to ITRF by the rigorous IAU 2006/2000A CIO chain — precession, nutation, Earth
/// rotation and polar motion. UT1 drives the Earth Rotation Angle; TT drives precession-nutation and
/// the TIO locator. Unlike [`teme_to_itrf`], the pole is tracked exactly rather than approximated by a
/// fixed axis; this is the frame the geopotential must be evaluated in for the tesseral terms to be
/// correct. Epochs are integer nanoseconds since J2000 on each scale; `xp`/`yp` in radians; `lod` in
/// seconds. The CIP series and frame chain are validated against pyerfa to < 5e-14.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // state, epoch on two scales, and the five EOP quantities
fn gcrf_to_itrf(
    r_gcrf: [f64; 3],
    v_gcrf: [f64; 3],
    ut1_ns: i128,
    tt_ns: i128,
    xp_rad: f64,
    yp_rad: f64,
    dx_rad: f64,
    dy_rad: f64,
    lod_seconds: f64,
) -> ([f64; 3], [f64; 3]) {
    let ut1 = Epoch::<Ut1>::from_ns_since_j2000(ut1_ns);
    let tt = Epoch::<Tt>::from_ns_since_j2000(tt_ns);
    let tf = frames_gcrf_to_itrf(
        ut1,
        tt,
        Angle::from_radians(xp_rad),
        Angle::from_radians(yp_rad),
        Angle::from_radians(dx_rad),
        Angle::from_radians(dy_rad),
        lod_seconds,
    );
    let sv = tf.state(StateVector::<Gcrf>::new(vec3(r_gcrf), vec3(v_gcrf)));
    (arr(sv.position.vector), arr(sv.velocity.vector))
}

/// Two-body propagation with switchable precision: the same 8th-order symplectic method (`yoshida8`)
/// run in `f64` or in double-double (`precision="dd"`, ~32 significant digits). At a small step the
/// `f64` run is capped by its round-off floor while the double-double run keeps falling — trading ~2×
/// the arithmetic for accuracy below what any `f64` propagator reaches. Advances `(r, v)` by `steps`
/// fixed steps of `dt` seconds under a point mass `mu`; returns the final state (m, m/s).
#[pyfunction]
#[pyo3(signature = (r0, v0, mu, dt, steps, precision="f64"))]
fn propagate_two_body(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    mu: f64,
    dt: f64,
    steps: usize,
    precision: &str,
) -> PyResult<([f64; 3], [f64; 3])> {
    py.detach(|| match precision {
        "f64" => {
            let accel = |x: &[f64; 3]| {
                let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                let k = -mu / (r2 * r2.sqrt());
                [k * x[0], k * x[1], k * x[2]]
            };
            let mut x = r0;
            let mut v = v0;
            yoshida8_scalar(&accel, &mut x, &mut v, dt, steps);
            Ok((x, v))
        }
        "dd" => {
            let neg_mu = DoubleDouble::from_f64(-mu);
            let accel = |x: &[DoubleDouble; 3]| {
                let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                let k = neg_mu / (r2 * r2.sqrt());
                [k * x[0], k * x[1], k * x[2]]
            };
            let mut x = [
                DoubleDouble::from_f64(r0[0]),
                DoubleDouble::from_f64(r0[1]),
                DoubleDouble::from_f64(r0[2]),
            ];
            let mut v = [
                DoubleDouble::from_f64(v0[0]),
                DoubleDouble::from_f64(v0[1]),
                DoubleDouble::from_f64(v0[2]),
            ];
            yoshida8_scalar(&accel, &mut x, &mut v, dt, steps);
            Ok((
                [x[0].value(), x[1].value(), x[2].value()],
                [v[0].value(), v[1].value(), v[2].value()],
            ))
        }
        _ => Err(PyValueError::new_err("precision must be 'f64' or 'dd'")),
    })
}

/// Point mass + J2 with switchable precision — the same yoshida8 symplectic method in `f64` or
/// double-double, now on a **perturbed** orbit. In double-double the J2 propagation drops below the
/// `f64` round-off floor that caps every f64 tool (Orekit, GMAT, nyx), so a real-force trajectory is
/// carried to ~32 digits. Advances `(r, v)` by `steps` fixed steps of `dt` seconds; J2 is evaluated in
/// the inertial frame about a fixed z — a conservative, autonomous force, as the symplectic method needs.
#[pyfunction]
#[pyo3(signature = (r0, v0, mu, j2, re, dt, steps, precision="f64"))]
#[allow(clippy::too_many_arguments)] // state, two field constants, step, count, precision tag
fn propagate_j2_precision(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    mu: f64,
    j2: f64,
    re: f64,
    dt: f64,
    steps: usize,
    precision: &str,
) -> PyResult<([f64; 3], [f64; 3])> {
    let force = (PointMass { mu }, J2 { mu, re, j2 });
    py.detach(|| match precision {
        "f64" => {
            let accel = |x: &[f64; 3]| force.acceleration::<f64>(0.0, x, &[0.0; 3]);
            let mut x = r0;
            let mut v = v0;
            yoshida8_scalar(&accel, &mut x, &mut v, dt, steps);
            Ok((x, v))
        }
        "dd" => {
            let zero = [DoubleDouble::from_f64(0.0); 3];
            let accel = |x: &[DoubleDouble; 3]| {
                force.acceleration::<DoubleDouble>(DoubleDouble::from_f64(0.0), x, &zero)
            };
            let mut x = [
                DoubleDouble::from_f64(r0[0]),
                DoubleDouble::from_f64(r0[1]),
                DoubleDouble::from_f64(r0[2]),
            ];
            let mut v = [
                DoubleDouble::from_f64(v0[0]),
                DoubleDouble::from_f64(v0[1]),
                DoubleDouble::from_f64(v0[2]),
            ];
            yoshida8_scalar(&accel, &mut x, &mut v, dt, steps);
            Ok((
                [x[0].value(), x[1].value(), x[2].value()],
                [v[0].value(), v[1].value(), v[2].value()],
            ))
        }
        _ => Err(PyValueError::new_err("precision must be 'f64' or 'dd'")),
    })
}

/// ITRF (Earth-fixed) Cartesian position in metres to WGS-84 geodetic `(latitude_deg, longitude_deg,
/// height_m)` by Vermeille's closed form. Height is above the ellipsoid, not the geoid. This is the
/// subpoint step of a ground track: feed it the ITRF position from [`teme_to_itrf`]. Raises
/// `ValueError` only inside the 43 km geocentric singularity, which no orbiting object occupies.
#[pyfunction]
fn itrf_to_geodetic(r_itrf: [f64; 3]) -> PyResult<(f64, f64, f64)> {
    let g = earth_itrf_to_geodetic(Position::new(vec3(r_itrf)))
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok((
        g.latitude.radians().to_degrees(),
        g.longitude.radians().to_degrees(),
        g.height.metres(),
    ))
}

/// WGS-84 geodetic `(latitude_deg, longitude_deg, height_m)` to an ITRF Cartesian position in metres —
/// the exact inverse of [`itrf_to_geodetic`], for placing ground stations and cells.
#[pyfunction]
fn geodetic_to_itrf(latitude_deg: f64, longitude_deg: f64, height_m: f64) -> [f64; 3] {
    let p = earth_geodetic_to_itrf(Geodetic {
        latitude: Angle::from_radians(latitude_deg.to_radians()),
        longitude: Angle::from_radians(longitude_deg.to_radians()),
        height: Length::from_metres(height_m),
    });
    arr(p.vector)
}

/// SGP4 propagation of one TLE to many minute-offsets from its epoch — a satellite ephemeris table.
/// Returns `[(r, v), …]` in TEME (m, m/s), one per entry of `minutes`. The element set is parsed once.
#[pyfunction]
fn sgp4_propagate_batch(
    py: Python<'_>,
    line1: &str,
    line2: &str,
    minutes: Vec<f64>,
) -> PyResult<Vec<([f64; 3], [f64; 3])>> {
    // Parse and propagate off the GIL: the work is pure Rust and touches no Python
    // object, so releasing it lets a Python-level `ThreadPoolExecutor` fan-out over
    // many TLEs run these calls truly in parallel. Errors travel back as plain strings
    // and become `ValueError`s once the GIL is reacquired (a `PyErr` is not `Ungil`).
    py.detach(|| {
        let sat = Satellite::from_tle(line1, line2).map_err(|e| format!("invalid TLE: {e:?}"))?;
        minutes
            .iter()
            .map(|&mn| {
                sat.propagate_minutes(mn)
                    .map(|sv| (arr(sv.position.vector), arr(sv.velocity.vector)))
                    .map_err(|e| format!("SGP4 failed at {mn} min: {e:?}"))
            })
            .collect::<Result<Vec<_>, String>>()
    })
    .map_err(PyValueError::new_err)
}

// --- measurement models (SI radians/metres), state `sat = [x,y,z,vx,vy,vz]` in the station's frame ---

/// Range (m) from an Earth-fixed station at `station_ecef` to the satellite.
#[pyfunction]
fn range(sat: [f64; 6], station_ecef: [f64; 3]) -> f64 {
    m_range(&sat, &Station::fixed(station_ecef))
}

/// Range rate (m/s), positive when opening, from a fixed station to the satellite.
#[pyfunction]
fn range_rate(sat: [f64; 6], station_ecef: [f64; 3]) -> f64 {
    m_range_rate(&sat, &Station::fixed(station_ecef))
}

/// Topocentric elevation angle (rad) at a geodetic station `(station_ecef, latitude, longitude)`.
#[pyfunction]
fn elevation(sat: [f64; 6], station_ecef: [f64; 3], latitude: f64, longitude: f64) -> f64 {
    m_elevation(
        &sat,
        &Topocentric::from_geodetic(station_ecef, latitude, longitude),
    )
}

/// Topocentric azimuth angle (rad, from north through east) at a geodetic station.
#[pyfunction]
fn azimuth(sat: [f64; 6], station_ecef: [f64; 3], latitude: f64, longitude: f64) -> f64 {
    m_azimuth(
        &sat,
        &Topocentric::from_geodetic(station_ecef, latitude, longitude),
    )
}

/// A 6-state square-root information filter (orbit-determination least squares by orthogonal
/// factorisation — never the normal equations). Accumulate measurement rows, then `solve`.
#[pyclass]
struct Srif6 {
    inner: Srif<6>,
}

#[pymethods]
impl Srif6 {
    /// An empty (diffuse-prior) filter.
    #[new]
    fn new() -> Self {
        Self {
            inner: Srif::default(),
        }
    }

    /// A filter seeded with a prior estimate `x0` and its information square-root `info_sqrt`.
    #[staticmethod]
    fn with_prior(x0: [f64; 6], info_sqrt: [[f64; 6]; 6]) -> Self {
        Self {
            inner: Srif::with_prior(&x0, &info_sqrt),
        }
    }

    /// Fold in one scalar measurement: partials `h` (∂z/∂x), observed-minus-computed `z`, and `weight`
    /// (`1/σ²`).
    fn add_row(&mut self, h: [f64; 6], z: f64, weight: f64) {
        self.inner.add_row(&h, z, weight);
    }

    /// The least-squares state estimate. Raises `ValueError` if the system is rank-deficient.
    fn solve(&self) -> PyResult<[f64; 6]> {
        self.inner
            .solve()
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))
    }

    /// The estimate covariance. Raises `ValueError` if rank-deficient.
    fn covariance(&self) -> PyResult<[[f64; 6]; 6]> {
        self.inner
            .covariance()
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))
    }

    /// `(sum of squared weighted residuals, row count)`.
    fn residual_stats(&self) -> (f64, u32) {
        self.inner.residual_stats()
    }
}

/// A dense ephemeris: an accepted-step interpolant of a propagated orbit, queryable at any time in the
/// span without re-integrating. Built by [`propagate_dense`].
#[pyclass]
struct DenseEphemeris {
    traj: Trajectory<6, Dop853Dense<6>>,
}

#[pymethods]
impl DenseEphemeris {
    /// State `(r, v)` at `t` seconds from the start (interpolated).
    fn eval(&self, t: f64) -> ([f64; 3], [f64; 3]) {
        let y = self.traj.eval(t);
        ([y[0], y[1], y[2]], [y[3], y[4], y[5]])
    }

    /// State `(r, v)` at the end of the span.
    fn final_state(&self) -> ([f64; 3], [f64; 3]) {
        let y = self.traj.final_state();
        ([y[0], y[1], y[2]], [y[3], y[4], y[5]])
    }

    /// `(t0, t1)` of the span, seconds.
    fn span(&self) -> (f64, f64) {
        self.traj.span()
    }

    /// `(accepted, rejected)` step counts — how hard the problem was.
    fn step_counts(&self) -> (u32, u32) {
        self.traj.step_counts()
    }
}

/// A machine-precision dense ephemeris from [`propagate_ias15_dense`], queryable anywhere on its span.
#[pyclass]
struct Ias15Ephemeris {
    traj: Trajectory<6, Ias15Dense<3>>,
}

#[pymethods]
impl Ias15Ephemeris {
    /// State `(r, v)` at `t` seconds from the start (interpolated).
    fn eval(&self, t: f64) -> ([f64; 3], [f64; 3]) {
        let y = self.traj.eval(t);
        ([y[0], y[1], y[2]], [y[3], y[4], y[5]])
    }

    /// State `(r, v)` at the end of the span.
    fn final_state(&self) -> ([f64; 3], [f64; 3]) {
        let y = self.traj.final_state();
        ([y[0], y[1], y[2]], [y[3], y[4], y[5]])
    }

    /// `(t0, t1)` of the span, seconds.
    fn span(&self) -> (f64, f64) {
        self.traj.span()
    }

    /// `(accepted, rejected)` step counts.
    fn step_counts(&self) -> (u32, u32) {
        self.traj.step_counts()
    }
}

/// Propagate `(r0, v0)` (m, m/s) for `t1` seconds under point-mass gravity `mu` plus `J₂` (`re`
/// reference radius) with the 15th-order Gauss–Radau **IAS15** integrator (Everhart / Rein–Spiegel) and
/// its compensated summation — reaching the f64 round-off floor (nanometres over a day), an order of
/// magnitude below a DOP853's micrometres. Position-only forces, so no drag; set `j2 = 0` for pure
/// two-body. Returns an [`Ias15Ephemeris`]. Raises `ValueError` on a failed integration.
#[pyfunction]
#[pyo3(signature = (r0, v0, t1, mu, j2 = 0.0, re = 6_378_137.0))]
fn propagate_ias15_dense(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    t1: f64,
    mu: f64,
    j2: f64,
    re: f64,
) -> PyResult<Ias15Ephemeris> {
    let forces = (PointMass { mu }, J2 { mu, re, j2 });
    let accel = |r: &[f64; 3]| forces.acceleration::<f64>(0.0, r, &[0.0_f64; 3]);
    let dt0 = (t1 / 1000.0).abs().max(1.0);
    let traj = py
        .detach(|| {
            integrate_ias15_dense::<3, 6, _>(&accel, &r0, &v0, 0.0, t1, dt0, DEFAULT_MAX_STEPS)
        })
        .map_err(|e| PyValueError::new_err(format!("IAS15 propagation failed: {e:?}")))?;
    Ok(Ias15Ephemeris { traj })
}

/// Propagate `(r0, v0)` (m, m/s) for `t1` seconds under point-mass gravity `mu` plus `J₂` oblateness
/// (`re` reference radius), adapting the DOP853 step to `tol`, and return a queryable
/// [`DenseEphemeris`]. Set `j2 = 0` for pure two-body. Raises `ValueError` on a failed integration.
#[pyfunction]
#[pyo3(signature = (r0, v0, t1, mu, j2 = 0.0, re = 6_378_137.0, tol = 1e-10))]
#[allow(clippy::too_many_arguments)] // GIL token, state, span, and the field constants
fn propagate_dense(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    t1: f64,
    mu: f64,
    j2: f64,
    re: f64,
    tol: f64,
) -> PyResult<DenseEphemeris> {
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let sys = Plain(Newtonian((PointMass { mu }, J2 { mu, re, j2 })));
    let h0 = (t1 / 1000.0).abs().max(1.0);
    let traj = py
        .detach(|| {
            integrate_dop853(
                &sys,
                0.0,
                &y0,
                t1,
                h0,
                Controls::new(Tolerance::new(tol, tol), 1e-3),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("dense propagation failed: {e:?}")))?;
    Ok(DenseEphemeris { traj })
}

/// A Python residual model for [`gauss_newton`]: wraps a callable `f(x) -> [(h, residual, weight), …]`.
/// The GIL is held for the whole `solve`, so calling back into Python is safe; any Python error is
/// captured and re-raised after `solve` returns.
struct PyResidualModel<'py> {
    func: &'py Bound<'py, PyAny>,
    err: core::cell::RefCell<Option<PyErr>>,
}

impl ResidualModel<6> for PyResidualModel<'_> {
    fn residuals(&self, x: &[f64; 6], emit: &mut dyn FnMut(&[f64; 6], f64, f64)) {
        if self.err.borrow().is_some() {
            return;
        }
        let rows = self
            .func
            .call1((x.to_vec(),))
            .and_then(|r| r.extract::<Vec<([f64; 6], f64, f64)>>());
        match rows {
            Ok(rows) => {
                for (h, residual, weight) in rows {
                    emit(&h, residual, weight);
                }
            }
            Err(e) => *self.err.borrow_mut() = Some(e),
        }
    }
}

/// The 6×6 formal error covariance of a solved 6-state (row-major); its diagonal square-roots are the
/// 1σ uncertainties.
type Covariance6 = [[f64; 6]; 6];

/// Gauss–Newton / Levenberg–Marquardt least squares for a 6-state, driven by a Python residual model
/// `residual_model(x) -> [(h, residual, weight), …]` called at each iterate, where each row is one
/// scalar measurement: `residual = computed(x) − observed`, `h = ∂residual/∂x` (length 6), `weight =
/// 1/σ²`. Returns `(state, covariance, converged, iterations, cost)`, where `covariance` is the 6×6
/// formal error covariance at the solution (its diagonal square-roots are the 1σ uncertainties). Raises
/// whatever the model raises, or `ValueError` on a rank-deficient solve.
#[pyfunction]
fn gauss_newton(
    residual_model: &Bound<'_, PyAny>,
    x0: [f64; 6],
) -> PyResult<([f64; 6], Covariance6, bool, u32, f64)> {
    let model = PyResidualModel {
        func: residual_model,
        err: core::cell::RefCell::new(None),
    };
    let solved = GaussNewton::default().solve(&model, &x0);
    if let Some(e) = model.err.into_inner() {
        return Err(e);
    }
    let (x, cov, report) = solved.map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
    Ok((x, cov, report.converged, report.iterations, report.cost))
}

/// Like [`propagate_dense`] but under the full tesseral geopotential (embedded XGM2019e to `degree`),
/// evaluated Earth-fixed at each step (rotation `θ = θ₀ + ω·t`, ω = Earth's rate) so the longitude-
/// dependent terms are correct. Returns a [`DenseEphemeris`]. Slower at high degree; keep `degree`
/// modest for long arcs.
#[pyfunction]
#[pyo3(signature = (r0, v0, t1, degree, theta0 = 0.0, tol = 1e-9))]
fn propagate_geopotential_dense(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    t1: f64,
    degree: usize,
    theta0: f64,
    tol: f64,
) -> PyResult<DenseEphemeris> {
    let geo = GeoPotential::embedded(degree)
        .ok_or_else(|| PyValueError::new_err("degree exceeds the embedded model (max 70)"))?;
    let sys = Plain(Newtonian(EarthFixed {
        inner: geo,
        omega: EARTH_ROTATION_RATE,
        theta0,
    }));
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let h0 = (t1 / 1000.0).abs().max(1.0);
    let traj = py
        .detach(|| {
            integrate_dop853(
                &sys,
                0.0,
                &y0,
                t1,
                h0,
                Controls::new(Tolerance::new(tol, tol), 1e-3),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("dense propagation failed: {e:?}")))?;
    Ok(DenseEphemeris { traj })
}

/// Build the fast, interpolated GCRF→ITRF provider ([`spacelab_frames::CioInterpolator`]) from the raw
/// ns/rad bridge inputs — shared by the fixed-EOP ITRF propagators. The CIP series (~1600 nutation
/// terms, ~40 µs) is gridded once and Lagrange-interpolated per stage, applying the exact ERA; the
/// result matches the per-stage `frames_gcrf_to_itrf` to the f64 floor (sub-µm at LEO).
#[allow(clippy::too_many_arguments)] // two epochs, the span, and the four fixed EOP scalars
fn cio_fast(
    epoch_tt_ns: i128,
    epoch_ut1_ns: i128,
    span: f64,
    xp_rad: f64,
    yp_rad: f64,
    dx_rad: f64,
    dy_rad: f64,
) -> CioInterpolator {
    CioInterpolator::new(
        Epoch::<Ut1>::from_ns_since_j2000(epoch_ut1_ns),
        Epoch::<Tt>::from_ns_since_j2000(epoch_tt_ns),
        span,
        Angle::from_radians(xp_rad),
        Angle::from_radians(yp_rad),
        Angle::from_radians(dx_rad),
        Angle::from_radians(dy_rad),
        0.0,
    )
}

/// The geodetic point the drag atmospheres read at time `t`: the inertial position rotated into the
/// ITRF, then to WGS-84 as `(altitude km, latitude deg, longitude deg)`, the triple both atmosphere
/// closures take.
fn geodetic_over(cio: CioInterpolator) -> impl Fn(f64, [f64; 3]) -> (f64, f64, f64) {
    move |t: f64, rf: [f64; 3]| {
        let m = cio.rotation_rows(t);
        let r_itrf = [
            m[0][0] * rf[0] + m[0][1] * rf[1] + m[0][2] * rf[2],
            m[1][0] * rf[0] + m[1][1] * rf[1] + m[1][2] * rf[2],
            m[2][0] * rf[0] + m[2][1] * rf[1] + m[2][2] * rf[2],
        ];
        match earth_itrf_to_geodetic(Position::new(vec3(r_itrf))) {
            Ok(g) => (
                g.height.metres() / 1000.0,
                g.latitude.degrees(),
                g.longitude.degrees(),
            ),
            Err(_) => (0.0, 0.0, 0.0),
        }
    }
}

/// Like [`propagate_geopotential_dense`], but the geopotential is evaluated in the **rigorous ITRF**
/// (IAU 2006/2000A CIO: precession, nutation, ERA, polar motion) rather than a fixed-axis rotation.
/// The pole is tracked exactly, which is what an Earth-fixed field actually requires: the fixed-axis
/// approximation drifts from the true pole by the accumulated precession — about 1.5 km a day at
/// present epochs. This is the Orekit-parity path; prefer it over [`propagate_geopotential_dense`]
/// when absolute accuracy in the Earth frame matters.
///
/// Epochs are integer nanoseconds since J2000; TT drives precession-nutation, UT1 drives ERA. `xp`/`yp`
/// (radians) and `lod` (seconds) are the polar-motion and length-of-day EOP, defaulting to zero.
#[pyfunction]
#[pyo3(signature = (
    r0, v0, t1, degree, epoch_tt_ns, epoch_ut1_ns,
    xp_rad = 0.0, yp_rad = 0.0, dx_rad = 0.0, dy_rad = 0.0, lod_seconds = 0.0, tol = 1e-9,
))]
#[allow(clippy::too_many_arguments)] // state, span, field degree, epoch on two scales, and the EOP triple
fn propagate_geopotential_itrf_dense(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    t1: f64,
    degree: usize,
    epoch_tt_ns: i128,
    epoch_ut1_ns: i128,
    xp_rad: f64,
    yp_rad: f64,
    dx_rad: f64,
    dy_rad: f64,
    lod_seconds: f64,
    tol: f64,
) -> PyResult<DenseEphemeris> {
    let geo = GeoPotential::embedded(degree)
        .ok_or_else(|| PyValueError::new_err("degree exceeds the embedded model (max 70)"))?;
    let _ = lod_seconds; // the CIO rotation is independent of LOD (it only scales the discarded ω).
    let cio = cio_fast(
        epoch_tt_ns,
        epoch_ut1_ns,
        t1,
        xp_rad,
        yp_rad,
        dx_rad,
        dy_rad,
    );
    let rotate = move |t: f64| cio.rotation_rows(t);
    let sys = Plain(Newtonian(BodyOriented { inner: geo, rotate }));
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let h0 = (t1 / 1000.0).abs().max(1.0);
    let traj = py
        .detach(|| {
            integrate_dop853(
                &sys,
                0.0,
                &y0,
                t1,
                h0,
                Controls::new(Tolerance::new(tol, tol), 1e-3),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("dense propagation failed: {e:?}")))?;
    Ok(DenseEphemeris { traj })
}

/// Point mass + J2 with the J2 bulge evaluated in the rigorous ITRF, taking explicit field constants
/// (`mu`, `J₂`, `Rₑ`) — the matched-field counterpart of [`propagate_geopotential_itrf_dense`] for
/// scoring against an external tool that supplies its own constants. The point mass is frame-invariant
/// so only J2 is rotated; its symmetry axis is the true (precessing, nutating) pole, not a fixed one.
/// Epochs are ns since J2000 (TT for precession-nutation, UT1 for ERA); `xp`/`yp` rad, `lod` s.
#[pyfunction]
#[pyo3(signature = (
    r0, v0, t1, mu, j2, re, epoch_tt_ns, epoch_ut1_ns,
    xp_rad = 0.0, yp_rad = 0.0, dx_rad = 0.0, dy_rad = 0.0, lod_seconds = 0.0, tol = 1e-11,
))]
#[allow(clippy::too_many_arguments)] // state, span, three field constants, epoch on two scales, EOP triple
fn propagate_j2_itrf_dense(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    t1: f64,
    mu: f64,
    j2: f64,
    re: f64,
    epoch_tt_ns: i128,
    epoch_ut1_ns: i128,
    xp_rad: f64,
    yp_rad: f64,
    dx_rad: f64,
    dy_rad: f64,
    lod_seconds: f64,
    tol: f64,
) -> PyResult<DenseEphemeris> {
    let _ = lod_seconds; // the CIO rotation is independent of LOD (it only scales the discarded ω).
    let cio = cio_fast(
        epoch_tt_ns,
        epoch_ut1_ns,
        t1,
        xp_rad,
        yp_rad,
        dx_rad,
        dy_rad,
    );
    let rotate = move |t: f64| cio.rotation_rows(t);
    let sys = Plain(Newtonian((
        PointMass { mu },
        BodyOriented {
            inner: J2 { mu, re, j2 },
            rotate,
        },
    )));
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let h0 = (t1 / 1000.0).abs().max(1.0);
    let traj = py
        .detach(|| {
            integrate_dop853(
                &sys,
                0.0,
                &y0,
                t1,
                h0,
                Controls::new(Tolerance::new(tol, tol), 1e-3),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("dense propagation failed: {e:?}")))?;
    Ok(DenseEphemeris { traj })
}

/// Point mass + J2 in the rigorous ITRF with a **time-varying** EOP series — the micron-parity path.
/// `propagate_j2_itrf_dense` holds the EOP fixed at the epoch, which over a day leaves a few-millimetre
/// residual as the true pole drifts; this interpolates `xp, yp, dUT1, dX, dY` linearly from a supplied
/// series (`eop_t` seconds from the epoch), so the frame tracks the pole exactly as the reference does.
/// `epoch_utc_ns` is the UTC epoch (ns since J2000, leap-second-naive); `tt_minus_utc_s` = ΔAT + 32.184.
#[pyfunction]
#[pyo3(signature = (
    r0, v0, t1, mu, j2, re, epoch_utc_ns, tt_minus_utc_s,
    eop_t, eop_xp, eop_yp, eop_dut1, eop_dx, eop_dy, tol = 1e-11,
))]
#[allow(clippy::too_many_arguments)] // state, span, field constants, time base, and the six EOP series
fn propagate_j2_itrf_eop_dense(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    t1: f64,
    mu: f64,
    j2: f64,
    re: f64,
    epoch_utc_ns: i128,
    tt_minus_utc_s: f64,
    eop_t: Vec<f64>,
    eop_xp: Vec<f64>,
    eop_yp: Vec<f64>,
    eop_dut1: Vec<f64>,
    eop_dx: Vec<f64>,
    eop_dy: Vec<f64>,
    tol: f64,
) -> PyResult<DenseEphemeris> {
    let n = eop_t.len();
    if n < 2
        || [&eop_xp, &eop_yp, &eop_dut1, &eop_dx, &eop_dy]
            .iter()
            .any(|s| s.len() != n)
    {
        return Err(PyValueError::new_err(
            "EOP series arrays must have equal length >= 2",
        ));
    }
    let rotate = move |t: f64| {
        let xp = lerp_at(&eop_t, &eop_xp, t);
        let yp = lerp_at(&eop_t, &eop_yp, t);
        let dut1 = lerp_at(&eop_t, &eop_dut1, t);
        let dx = lerp_at(&eop_t, &eop_dx, t);
        let dy = lerp_at(&eop_t, &eop_dy, t);
        let tt = Epoch::<Tt>::from_ns_since_j2000(
            epoch_utc_ns + ((t + tt_minus_utc_s) * 1e9).round() as i128,
        );
        let ut1 =
            Epoch::<Ut1>::from_ns_since_j2000(epoch_utc_ns + ((t + dut1) * 1e9).round() as i128);
        frames_gcrf_to_itrf(
            ut1,
            tt,
            Angle::from_radians(xp),
            Angle::from_radians(yp),
            Angle::from_radians(dx),
            Angle::from_radians(dy),
            0.0,
        )
        .rotation
        .matrix()
        .rows
    };
    let sys = Plain(Newtonian((
        PointMass { mu },
        BodyOriented {
            inner: J2 { mu, re, j2 },
            rotate,
        },
    )));
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let h0 = (t1 / 1000.0).abs().max(1.0);
    let traj = py
        .detach(|| {
            integrate_dop853(
                &sys,
                0.0,
                &y0,
                t1,
                h0,
                Controls::new(Tolerance::new(tol, tol), 1e-3),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("dense propagation failed: {e:?}")))?;
    Ok(DenseEphemeris { traj })
}

/// Point mass + tesseral geopotential + the IERS 2010 solid Earth tide, all in the rigorous ITRF: the
/// geopotential plus the Sun/Moon-raised tidal deformation of the low-degree field (a ~cm–dm effect on
/// a LEO orbit over a day). Sun and Moon come from the built-in analytic ephemerides at
/// `epoch_j2000_seconds`; `mu_sun`/`mu_moon` set the tidal amplitudes. Epochs are ns since J2000.
#[pyfunction]
#[pyo3(signature = (
    r0, v0, t1, degree, epoch_tt_ns, epoch_ut1_ns, epoch_j2000_seconds,
    xp_rad = 0.0, yp_rad = 0.0, dx_rad = 0.0, dy_rad = 0.0, lod_seconds = 0.0,
    mu_sun = 1.327_124_400_18e20, mu_moon = 4.902_800_066e12, tol = 1e-9,
))]
#[allow(clippy::too_many_arguments)] // state, span, field degree, epoch on three scales, EOP, two GM
fn propagate_geopotential_tide_dense(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    t1: f64,
    degree: usize,
    epoch_tt_ns: i128,
    epoch_ut1_ns: i128,
    epoch_j2000_seconds: f64,
    xp_rad: f64,
    yp_rad: f64,
    dx_rad: f64,
    dy_rad: f64,
    lod_seconds: f64,
    mu_sun: f64,
    mu_moon: f64,
    tol: f64,
) -> PyResult<DenseEphemeris> {
    let geo = GeoPotential::embedded(degree)
        .ok_or_else(|| PyValueError::new_err("degree exceeds the embedded model (max 70)"))?;
    let _ = lod_seconds; // the CIO rotation is independent of LOD (it only scales the discarded ω).
    // The tide force needs the frame twice (the geopotential's `BodyOriented` and the tide's own
    // `rotate`), so share one grid via `Arc` — cheap-clone, and `Send`/`Sync` so `sys` still crosses
    // `Python::detach`.
    let cio = std::sync::Arc::new(cio_fast(
        epoch_tt_ns,
        epoch_ut1_ns,
        t1,
        xp_rad,
        yp_rad,
        dx_rad,
        dy_rad,
    ));
    let cio_tide = std::sync::Arc::clone(&cio);
    let rotate = move |t: f64| cio.rotation_rows(t);
    let rotate_tide = move |t: f64| cio_tide.rotation_rows(t);
    let tide = SolidEarthTide {
        mu_earth: 3.986_004_415e14, // MODEL_GM of the embedded geopotential
        re_earth: 6_378_136.3,      // MODEL_RE
        mu_sun,
        mu_moon,
        sun: AnalyticSun {
            epoch_j2000_seconds,
        },
        moon: AnalyticMoon {
            epoch_j2000_seconds,
        },
        rotate: rotate_tide,
        // Full IERS 2010 degree-2 tide each step: frequency-dependent Step 2 (GMST + Delaunay from
        // the TT/UT1 timeline) plus the solid + ocean pole tides (from the epoch polar motion).
        step2: move |t: f64| {
            let dt_ns = (t * 1e9).round() as i128;
            let tt = Epoch::<Tt>::from_ns_since_j2000(epoch_tt_ns + dt_ns);
            let ut1 = Epoch::<Ut1>::from_ns_since_j2000(epoch_ut1_ns + dt_ns);
            let gmst = spacelab_fa::gmst82(ut1).radians();
            let tc = tt.julian_centuries();
            let delaunay = [
                spacelab_fa::fundamental::fal03(tc),
                spacelab_fa::fundamental::falp03(tc),
                spacelab_fa::fundamental::faf03(tc),
                spacelab_fa::fundamental::fad03(tc),
                spacelab_fa::fundamental::faom03(tc),
            ];
            let t_j2000 = (epoch_tt_ns + dt_ns) as f64 / 1e9; // seconds since J2000 (TT)
            let mut s = spacelab_propagation::step2_corrections(gmst, delaunay);
            let pole = spacelab_propagation::MeanPole::Secular2018;
            let sp = spacelab_propagation::pole_tide_corrections(xp_rad, yp_rad, t_j2000, pole);
            let op =
                spacelab_propagation::ocean_pole_tide_corrections(xp_rad, yp_rad, t_j2000, pole);
            s[1] += sp[0] + op[0]; // ΔC̄21
            s[2] += sp[1] + op[1]; // ΔS̄21
            s
        },
    };
    let sys = Plain(Newtonian((BodyOriented { inner: geo, rotate }, tide)));
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let h0 = (t1 / 1000.0).abs().max(1.0);
    let traj = py
        .detach(|| {
            integrate_dop853(
                &sys,
                0.0,
                &y0,
                t1,
                h0,
                Controls::new(Tolerance::new(tol, tol), 1e-3),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("dense propagation failed: {e:?}")))?;
    Ok(DenseEphemeris { traj })
}

/// Point mass + the three IERS 2010 relativistic corrections (Conventions Ch. 10, Eq. 10.12), all in
/// the GCRF: Schwarzschild (the dominant term, a ~metre/day relativistic perigee precession),
/// Lense–Thirring frame dragging, and de Sitter geodesic precession. The Sun (for de Sitter) comes
/// from the built-in analytic ephemeris at `epoch_j2000_seconds`, its velocity by central difference.
/// Compare the result against a two-body ([`kepler_propagate`]) run to isolate the relativistic drift.
#[pyfunction]
#[pyo3(signature = (
    r0, v0, t1, epoch_j2000_seconds,
    mu = 3.986_004_418e14, mu_sun = 1.327_124_400_18e20, tol = 1e-9,
))]
#[allow(clippy::too_many_arguments)] // GIL token, state, span, epoch, and two gravitational parameters
fn propagate_relativistic_dense(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    t1: f64,
    epoch_j2000_seconds: f64,
    mu: f64,
    mu_sun: f64,
    tol: f64,
) -> PyResult<DenseEphemeris> {
    // Geocentric Sun position from the analytic ephemeris; its velocity by central difference (the
    // de Sitter term is ~4e-11 m/s², so a 60 s stencil is far more accurate than it needs to be).
    let sun_pv = move |t: f64| {
        let sun = AnalyticSun {
            epoch_j2000_seconds,
        };
        let dt = 60.0;
        let pm = sun.position(t - dt);
        let pp = sun.position(t + dt);
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
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let h0 = (t1 / 1000.0).abs().max(1.0);
    let traj = py
        .detach(|| {
            integrate_dop853(
                &sys,
                0.0,
                &y0,
                t1,
                h0,
                Controls::new(Tolerance::new(tol, tol), 1e-3),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("dense propagation failed: {e:?}")))?;
    Ok(DenseEphemeris { traj })
}

/// NRLMSISE-00 effective total mass density for drag (kg/m³) — the `gtd7d` output `d[5]`, including
/// anomalous oxygen — at a geodetic point and epoch. `doy` is the day of year, `sec` the UT seconds
/// into the day, `alt_km` the geodetic altitude, `lat_deg`/`lon_deg` the geodetic latitude/longitude;
/// `f107`/`f107a`/`ap` are the space-weather drivers. Local solar time is derived as `sec/3600 +
/// lon/15`. This is the raw model, exposed for validation against reference implementations.
#[pyfunction]
#[pyo3(signature = (doy, sec, alt_km, lat_deg, lon_deg, f107 = 150.0, f107a = 150.0, ap = 4.0))]
#[allow(clippy::too_many_arguments)] // day of year, UT, three geodetic coordinates, three space-weather
fn nrlmsise00_density(
    doy: i32,
    sec: f64,
    alt_km: f64,
    lat_deg: f64,
    lon_deg: f64,
    f107: f64,
    f107a: f64,
    ap: f64,
) -> f64 {
    let input = Nrlmsise00Input {
        year: 0,
        doy,
        sec,
        alt: alt_km,
        g_lat: lat_deg,
        g_long: lon_deg,
        lst: sec / 3600.0 + lon_deg / 15.0,
        f107a,
        f107,
        ap,
        ap_array: None,
    };
    gtd7d(&input, &Nrlmsise00Flags::default()).d[5] * 1000.0 // g/cm³ → kg/m³
}

/// **SDTM-1** total mass density (kg/m³) at a geodetic point and epoch — this library's own
/// thermosphere model, fitted to open accelerometer densities because the two models that measure
/// better cannot be redistributed. `doy` is the day of year, `sec` the UT seconds into the day,
/// `alt_km` the geodetic altitude, `lat_deg`/`lon_deg` the geodetic latitude/longitude. Local solar
/// time is derived as `sec/3600 + lon/15`, wrapped into `[0, 24)`.
///
/// `f107` is the **previous day's** observed 10.7 cm flux and `f107a` its 81-day **trailing** mean —
/// the trailing one, not the centred mean NRLMSISE-00 specifies, because the fit used the window that
/// stays an observation to the last published day. `storm` is the storm state's response over the
/// `ap` history, from `sdtm1_storm_response`; it is not an `ap` value and passing one is a different
/// model.
#[pyfunction]
#[pyo3(signature = (doy, sec, alt_km, lat_deg, lon_deg, f107 = 150.0, f107a = 150.0, storm = 0.0))]
#[allow(clippy::too_many_arguments)] // day of year, UT, three geodetic coordinates, three drivers
fn sdtm1_density(
    doy: f64,
    sec: f64,
    alt_km: f64,
    lat_deg: f64,
    lon_deg: f64,
    f107: f64,
    f107a: f64,
    storm: f64,
) -> f64 {
    SDTM1_FIELD.density(
        &sdtm1_point(doy, sec, alt_km, lat_deg, lon_deg),
        &SdtmDrivers { f107, f107a, storm },
    )
}

/// **SDTM-1** exospheric temperature `T∞` (K) at the same point and drivers as `sdtm1_density`.
///
/// The quantity the fit actually parameterises — the density is a Bates profile above it — so this is
/// the handle for comparing the model against a published exospheric temperature rather than against
/// a density, and for seeing the diurnal bulge and the storm response without the altitude profile on
/// top of them.
#[pyfunction]
#[pyo3(signature = (doy, sec, alt_km, lat_deg, lon_deg, f107 = 150.0, f107a = 150.0, storm = 0.0))]
#[allow(clippy::too_many_arguments)] // day of year, UT, three geodetic coordinates, three drivers
fn sdtm1_exospheric_temperature(
    doy: f64,
    sec: f64,
    alt_km: f64,
    lat_deg: f64,
    lon_deg: f64,
    f107: f64,
    f107a: f64,
    storm: f64,
) -> f64 {
    SDTM1_FIELD.exospheric_temperature(
        &sdtm1_point(doy, sec, alt_km, lat_deg, lon_deg),
        &SdtmDrivers { f107, f107a, storm },
    )
}

/// **SDTM-1**'s geomagnetic storm driver: the shipped heating and nitric-oxide cooling state run
/// over an index history.
///
/// `history` is the `ap` series oldest first, ending at the interval **before** the sample, and
/// `cadence_s` is that series' step — 10 800 s for CelesTrak's 3-hourly `ap`, which is what the fit
/// used. The three constants belong to the fitted field, so a caller cannot build this driver
/// itself. Nitric oxide remembers a storm for days, and a truncated history reads as a different
/// storm: pass fifteen days, 120 intervals (`sdtm::coefficients::HISTORY` in Rust).
#[pyfunction]
#[pyo3(signature = (history, cadence_s = 10_800.0))]
fn sdtm1_storm_response(history: Vec<f64>, cadence_s: f64) -> f64 {
    SDTM1_FIELD.storm().response(&history, cadence_s)
}

/// **SDTM-1**'s log-normal density interval: `(lower, upper)` in kg/m³ about `density`.
///
/// The spread of `ln(ρ_model/ρ_measured)` was measured by flux × storm regime on the residual of a fit
/// with the mission family held out, so the interval is the model's error against measurement rather
/// than a formal propagation of parameter covariance. `z` is the standard-normal multiplier: 1.645 for
/// 90 %, 1.960 for 95 %, 2.576 for 99 %. Which regime applies is decided by `f107a` and `storm`, so
/// they must be the same values `density` was evaluated with.
#[pyfunction]
#[pyo3(signature = (density, f107 = 150.0, f107a = 150.0, storm = 0.0, z = 1.644_853_626_951_472_7))]
fn sdtm1_density_interval(density: f64, f107: f64, f107a: f64, storm: f64, z: f64) -> (f64, f64) {
    SDTM1_UNCERTAINTY.interval(density, &SdtmDrivers { f107, f107a, storm }, z)
}

/// **SDTM-1**'s fitted envelope, `((lowest_km, highest_km), (lowest_f107a, highest_f107a))`: the
/// geodetic altitude and the 81-day trailing mean flux its rows span, both ends included. Inside it
/// SDTM-1 is the recommended model; outside it no row tests the field, and NRLMSISE-00 is the
/// recommendation (`docs/verification.md` §10.28).
#[pyfunction]
fn sdtm1_envelope() -> ((f64, f64), (f64, f64)) {
    let Envelope { altitude, f107a } = SDTM1_ENVELOPE;
    (
        (*altitude.start() / 1000.0, *altitude.end() / 1000.0),
        (*f107a.start(), *f107a.end()),
    )
}

/// The evaluation point the three `sdtm1_*` point functions share, assembled the way the fit's own
/// data was: a fractional day of year, and local solar time from UT and longitude wrapped into a day.
fn sdtm1_point(doy: f64, sec: f64, alt_km: f64, lat_deg: f64, lon_deg: f64) -> SdtmPoint {
    let mut lst = sec / 3600.0 + lon_deg / 15.0;
    lst -= 24.0 * (lst / 24.0).floor();
    SdtmPoint {
        altitude: alt_km * 1000.0,
        latitude: lat_deg.to_radians(),
        local_solar_time: lst,
        day_of_year: doy + sec / 86_400.0,
    }
}

/// Point mass + atmospheric drag with the **NRLMSISE-00** thermospheric density model — the modern
/// standard for LEO drag, replacing the single-exponential atmosphere with a full empirical model of
/// temperature and species density driven by day of year, local solar time and space weather
/// (`f107`, `f107a`, `ap`). The satellite's geodetic altitude/latitude/longitude are taken in the
/// rigorous ITRF.
///
/// `ballistic_coefficient` is `C_d A / m` (m²/kg). `doy` (day of year) and `sec_of_day` (UT) set the
/// NRLMSISE epoch, advanced with the propagation. Epochs are ns since J2000. Returns a
/// [`DenseEphemeris`]; compare a drag run against a two-body run to see the decay.
#[pyfunction]
#[pyo3(signature = (
    r0, v0, t1, epoch_tt_ns, epoch_ut1_ns, doy, sec_of_day, ballistic_coefficient,
    f107 = 150.0, f107a = 150.0, ap = 4.0, mu = 3.986_004_418e14,
    xp_rad = 0.0, yp_rad = 0.0, dx_rad = 0.0, dy_rad = 0.0, lod_seconds = 0.0, tol = 1e-9,
))]
#[allow(clippy::too_many_arguments)] // state, span, two epochs, NRLMSISE epoch, BC, space weather, EOP
fn propagate_drag_nrlmsise00_dense(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    t1: f64,
    epoch_tt_ns: i128,
    epoch_ut1_ns: i128,
    doy: i32,
    sec_of_day: f64,
    ballistic_coefficient: f64,
    f107: f64,
    f107a: f64,
    ap: f64,
    mu: f64,
    xp_rad: f64,
    yp_rad: f64,
    dx_rad: f64,
    dy_rad: f64,
    lod_seconds: f64,
    tol: f64,
) -> PyResult<DenseEphemeris> {
    let _ = lod_seconds; // the CIO rotation is independent of LOD (it only scales the discarded ω).
    let geodetic = geodetic_over(cio_fast(
        epoch_tt_ns,
        epoch_ut1_ns,
        t1,
        xp_rad,
        yp_rad,
        dx_rad,
        dy_rad,
    ));
    let drag = Drag {
        ballistic_coefficient,
        earth_rotation_rate: EARTH_ROTATION_RATE,
        atmosphere: Nrlmsise00Atmosphere {
            doy0: doy,
            sec0: sec_of_day,
            f107,
            f107a,
            ap,
            geodetic,
        },
        wind: NoWind,
    };
    let sys = Plain(Newtonian((PointMass { mu }, drag)));
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let h0 = (t1 / 1000.0).abs().max(1.0);
    let traj = py
        .detach(|| {
            integrate_dop853(
                &sys,
                0.0,
                &y0,
                t1,
                h0,
                Controls::new(Tolerance::new(tol, tol), 1e-3),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("dense propagation failed: {e:?}")))?;
    Ok(DenseEphemeris { traj })
}

/// Point mass + atmospheric drag through **SDTM-1** — this library's own thermosphere model, and the
/// recommended one inside its fitted envelope, `sdtm1_envelope()`; outside it the recommendation is
/// NRLMSISE-00, `propagate_drag_nrlmsise00_dense` (`docs/verification.md` §10.28). The satellite's
/// geodetic altitude, latitude and longitude are taken in the rigorous ITRF, and the density between
/// two evaluations follows SDTM-1's own analytic scale height.
///
/// `ballistic_coefficient` is `C_d A / m` (m²/kg). `doy` (day of year) and `sec_of_day` (UT) set the
/// model's epoch, advanced with the propagation. `ap_history` is the `ap` series oldest first, ending
/// at the interval **before** the arc's epoch, one value every `cadence_s` seconds: the storm state is
/// run over it once, as `sdtm1_storm_response` runs it, and held over the arc with `f107`, the
/// previous day's flux, and `f107a`, its 81-day trailing mean. Nitric oxide remembers a storm for
/// days: pass fifteen days, 120 intervals. A history of one value is that value's steady state. An
/// empty history, a value that is not finite or a cadence that is not finite and positive raises
/// `ValueError`. An arc that leaves the envelope is integrated, not refused: there the model
/// extrapolates. Epochs are ns since J2000. Returns a [`DenseEphemeris`].
#[pyfunction]
#[pyo3(signature = (
    r0, v0, t1, epoch_tt_ns, epoch_ut1_ns, doy, sec_of_day, ballistic_coefficient, ap_history,
    f107 = 150.0, f107a = 150.0, cadence_s = 10_800.0, mu = 3.986_004_418e14,
    xp_rad = 0.0, yp_rad = 0.0, dx_rad = 0.0, dy_rad = 0.0, lod_seconds = 0.0, tol = 1e-9,
))]
#[allow(clippy::too_many_arguments)] // state, span, two epochs, SDTM-1 epoch, BC, space weather, EOP
fn propagate_drag_sdtm1_dense(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    t1: f64,
    epoch_tt_ns: i128,
    epoch_ut1_ns: i128,
    doy: f64,
    sec_of_day: f64,
    ballistic_coefficient: f64,
    ap_history: Vec<f64>,
    f107: f64,
    f107a: f64,
    cadence_s: f64,
    mu: f64,
    xp_rad: f64,
    yp_rad: f64,
    dx_rad: f64,
    dy_rad: f64,
    lod_seconds: f64,
    tol: f64,
) -> PyResult<DenseEphemeris> {
    let storm = sdtm1_drag_storm(&ap_history, cadence_s)?;
    let _ = lod_seconds; // the CIO rotation is independent of LOD (it only scales the discarded ω).
    let geodetic = geodetic_over(cio_fast(
        epoch_tt_ns,
        epoch_ut1_ns,
        t1,
        xp_rad,
        yp_rad,
        dx_rad,
        dy_rad,
    ));
    let drag = Drag {
        ballistic_coefficient,
        earth_rotation_rate: EARTH_ROTATION_RATE,
        atmosphere: SdtmAtmosphere {
            field: SDTM1_FIELD,
            drivers: SdtmDrivers { f107, f107a, storm },
            doy0: doy,
            sec0: sec_of_day,
            geodetic,
        },
        wind: NoWind,
    };
    let sys = Plain(Newtonian((PointMass { mu }, drag)));
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let h0 = (t1 / 1000.0).abs().max(1.0);
    let traj = py
        .detach(|| {
            integrate_dop853(
                &sys,
                0.0,
                &y0,
                t1,
                h0,
                Controls::new(Tolerance::new(tol, tol), 1e-3),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("dense propagation failed: {e:?}")))?;
    Ok(DenseEphemeris { traj })
}

/// The storm driver `propagate_drag_sdtm1_dense` holds over its arc, refused where the history
/// cannot give one: empty, a value that is not finite, or a cadence that is not finite and positive.
fn sdtm1_drag_storm(history: &[f64], cadence_s: f64) -> PyResult<f64> {
    if history.is_empty() {
        return Err(PyValueError::new_err(
            "ap_history is empty: pass fifteen days, 120 intervals",
        ));
    }
    if let Some(i) = history.iter().position(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "ap_history[{i}] is {}, and must be finite",
            history[i]
        )));
    }
    if !(cadence_s.is_finite() && cadence_s > 0.0) {
        return Err(PyValueError::new_err(format!(
            "cadence_s is {cadence_s}, and must be finite and positive"
        )));
    }
    Ok(SDTM1_FIELD.storm().response(history, cadence_s))
}

/// Like [`propagate_geopotential_dense`] but adding atmospheric drag — the dominant non-gravitational
/// LEO perturbation, and the one that decays the orbit. Gravity is the tesseral geopotential to
/// `degree` (Earth-fixed); drag opposes the velocity relative to the co-rotating atmosphere,
/// `a = −½ (C_d A/m) ρ |v_rel| v_rel`, with `ρ` a single-exponential atmosphere.
///
/// `ballistic_coefficient` is `C_d A / m` (m²/kg), a per-satellite property with no safe universal
/// default, so it is required. The atmosphere defaults to Vallado's 400 km reference
/// (`ρ₀ = 3.725×10⁻¹² kg/m³` at 400 km, scale height 58.515 km). Returns a [`DenseEphemeris`].
#[pyfunction]
#[pyo3(signature = (
    r0, v0, t1, degree, ballistic_coefficient,
    reference_density = 3.725e-12, reference_altitude = 400_000.0, scale_height = 58_515.0,
    earth_radius = 6_378_137.0, theta0 = 0.0, tol = 1e-9,
))]
#[allow(clippy::too_many_arguments)] // each argument is one physical constant of the field or atmosphere
fn propagate_drag_dense(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    t1: f64,
    degree: usize,
    ballistic_coefficient: f64,
    reference_density: f64,
    reference_altitude: f64,
    scale_height: f64,
    earth_radius: f64,
    theta0: f64,
    tol: f64,
) -> PyResult<DenseEphemeris> {
    let geo = GeoPotential::embedded(degree)
        .ok_or_else(|| PyValueError::new_err("degree exceeds the embedded model (max 70)"))?;
    let gravity = EarthFixed {
        inner: geo,
        omega: EARTH_ROTATION_RATE,
        theta0,
    };
    let drag = Drag {
        ballistic_coefficient,
        earth_rotation_rate: EARTH_ROTATION_RATE,
        atmosphere: ExponentialAtmosphere {
            reference_density,
            reference_altitude,
            scale_height,
            earth_radius,
        },
        wind: NoWind,
    };
    let sys = Plain(Newtonian((gravity, drag)));
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let h0 = (t1 / 1000.0).abs().max(1.0);
    let traj = py
        .detach(|| {
            integrate_dop853(
                &sys,
                0.0,
                &y0,
                t1,
                h0,
                Controls::new(Tolerance::new(tol, tol), 1e-3),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("dense propagation failed: {e:?}")))?;
    Ok(DenseEphemeris { traj })
}

/// Like [`propagate_geopotential_dense`] but adding the luni-solar third-body perturbations — the
/// forces that dominate above LEO, where the atmosphere is gone but the Sun and Moon steadily work the
/// orbit (this is what drives GEO nodal/inclination drift). Solar radiation pressure is included too,
/// off by default.
///
/// Sun and Moon positions come from the built-in low-precision analytic series, so no ephemeris kernel
/// is needed; `epoch_j2000_seconds` places `t = 0` on the timeline (seconds past J2000 TT). Set
/// `srp_reference_acceleration` to `Φ·C_r·A/m` (m/s²) to enable a cannonball SRP model with a
/// cylindrical Earth shadow. Returns a [`DenseEphemeris`].
#[pyfunction]
#[pyo3(signature = (
    r0, v0, t1, degree, epoch_j2000_seconds,
    srp_reference_acceleration = 0.0, theta0 = 0.0, tol = 1e-9,
))]
#[allow(clippy::too_many_arguments)] // each argument is one physical constant of the field, epoch, or SRP
fn propagate_luni_solar_dense(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    t1: f64,
    degree: usize,
    epoch_j2000_seconds: f64,
    srp_reference_acceleration: f64,
    theta0: f64,
    tol: f64,
) -> PyResult<DenseEphemeris> {
    // Heliocentric and lunar gravitational parameters (IAU 2015 / DE440) and the defining AU, SI.
    const MU_SUN: f64 = 1.327_124_400_18e20;
    const MU_MOON: f64 = 4.902_800_066e12;
    const ASTRONOMICAL_UNIT: f64 = 1.495_978_707e11;

    let geo = GeoPotential::embedded(degree)
        .ok_or_else(|| PyValueError::new_err("degree exceeds the embedded model (max 70)"))?;
    let gravity = EarthFixed {
        inner: geo,
        omega: EARTH_ROTATION_RATE,
        theta0,
    };
    let sun = AnalyticSun {
        epoch_j2000_seconds,
    };
    let moon = AnalyticMoon {
        epoch_j2000_seconds,
    };
    let sys = Plain(Newtonian((
        gravity,
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
            earth_radius: 6_378_137.0,
            sun,
        },
    )));
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let h0 = (t1 / 1000.0).abs().max(1.0);
    let traj = py
        .detach(|| {
            integrate_dop853(
                &sys,
                0.0,
                &y0,
                t1,
                h0,
                Controls::new(Tolerance::new(tol, tol), 1e-3),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("dense propagation failed: {e:?}")))?;
    Ok(DenseEphemeris { traj })
}

/// A loaded JPL DE ephemeris (e.g. DE440) backed by an ANISE SPK kernel. Query planetary/lunar states
/// by NAIF ID.
#[pyclass]
struct De440 {
    inner: De440Ephemeris,
}

#[pymethods]
impl De440 {
    /// Load an SPK kernel (e.g. a `de440s.bsp` path). Raises `ValueError` if it cannot be read.
    #[new]
    fn load(path: &str) -> PyResult<Self> {
        De440Ephemeris::load(path)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(format!("failed to load ephemeris: {e}")))
    }

    /// State `(r_km, v_km_s)` of `target` relative to `observer` (both NAIF IDs — 301 Moon, 399 Earth,
    /// 10 Sun, 499 Mars) at `et_seconds` (TDB seconds past J2000), in the J2000 frame.
    fn state(&self, target: i32, observer: i32, et_seconds: f64) -> PyResult<([f64; 3], [f64; 3])> {
        let bs = self
            .inner
            .state(
                EphFrame::from_ephem_j2000(target),
                EphFrame::from_ephem_j2000(observer),
                EphEpoch::from_et_seconds(et_seconds),
            )
            .map_err(|e| PyValueError::new_err(format!("ephemeris query failed: {e}")))?;
        Ok((arr(bs.position), arr(bs.velocity)))
    }
}

/// Brouwer–Lyddane analytical propagation under the zonal geopotential J2–J5. From an initial
/// osculating state `(r0, v0)` in an inertial frame, returns the osculating `(r, v)` at each time in
/// `times` (seconds from the epoch). `c20..c50 = −J2..−J5` are the un-normalized zonal coefficients.
/// `m2` (rad/s², like a TLE `B*`) is the empirical along-track secular term: with `m2 = 0` the theory
/// has a known in-track drift (~tens of km/day for a LEO, the orbit *plane* stays exact), fitted away
/// in orbit determination. Fast: a mean-element drift plus closed-form periodic terms, no integration.
#[pyfunction]
#[pyo3(signature = (
    r0, v0, times, mu = 3.986_004_415e14, re = 6_378_136.3,
    c20 = -1.082_626_68e-3, c30 = 2.532_656_49e-6, c40 = 1.619_621_59e-6, c50 = 2.272_960_83e-7,
    m2 = 0.0,
))]
#[allow(clippy::too_many_arguments)] // state, epochs, mu, radius, four zonals, empirical drag term
fn brouwer_lyddane_propagate(
    py: Python<'_>,
    r0: [f64; 3],
    v0: [f64; 3],
    times: Vec<f64>,
    mu: f64,
    re: f64,
    c20: f64,
    c30: f64,
    c40: f64,
    c50: f64,
    m2: f64,
) -> PyResult<Vec<([f64; 3], [f64; 3])>> {
    let osc = KeplerianElements::from_state(vec3(r0), vec3(v0), mu);
    let bl = BrouwerLyddane::from_osculating_with_drag(&osc, re, c20, c30, c40, c50, m2)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(py.detach(|| {
        times
            .iter()
            .map(|&t| {
                let (r, v) = bl.propagate(t);
                (arr(r), arr(v))
            })
            .collect()
    }))
}

/// DSST semi-analytical **mean-element** propagation under the zonal geopotential J2–Jn. Given an
/// initial *mean* equinoctial state `[a, k, h, q, p, λ]` (`k=e·cos ϖ`, `h=e·sin ϖ`, `q=tan(i/2)·cos Ω`,
/// `p=tan(i/2)·sin Ω`, `λ=M+ϖ`), returns the mean equinoctial state at each time in `times` (seconds).
/// `j` is `[J2, J3, …]`. The orbit-averaged rates are integrated with large steps — cheap over long
/// arcs, capturing the secular drift (node regression, apsidal precession) without short-period noise.
#[pyfunction]
#[pyo3(signature = (
    initial, times, mu = 3.986_004_415e14, re = 6_378_136.3,
    j = vec![1.082_626_68e-3, -2.532_656_49e-6, -1.619_621_59e-6, -2.272_960_83e-7],
))]
fn dsst_zonal_propagate_mean(
    py: Python<'_>,
    initial: [f64; 6],
    times: Vec<f64>,
    mu: f64,
    re: f64,
    j: Vec<f64>,
) -> Vec<[f64; 6]> {
    let zonal = DsstZonal::from_j(mu, re, &j);
    py.detach(|| {
        times
            .iter()
            .map(|&t| zonal.propagate_mean(initial, t))
            .collect()
    })
}

/// DSST **osculating** propagation under the zonal geopotential J2–Jn: the mean-element drift plus the
/// analytic short-period corrections. Given an initial *mean* equinoctial state `[a, k, h, q, p, λ]`,
/// returns the *osculating* equinoctial state at each time — the full DSST OSCULATING output.
#[pyfunction]
#[pyo3(signature = (
    initial, times, mu = 3.986_004_415e14, re = 6_378_136.3,
    j = vec![1.082_626_68e-3, -2.532_656_49e-6, -1.619_621_59e-6, -2.272_960_83e-7],
))]
fn dsst_zonal_propagate_osculating(
    py: Python<'_>,
    initial: [f64; 6],
    times: Vec<f64>,
    mu: f64,
    re: f64,
    j: Vec<f64>,
) -> Vec<[f64; 6]> {
    let zonal = DsstZonal::from_j(mu, re, &j);
    py.detach(|| {
        times
            .iter()
            .map(|&t| zonal.propagate_osculating(initial, t))
            .collect()
    })
}

/// IERS 2010 solid-tide Step-2 (frequency-dependent) corrections to the degree-2 geopotential
/// coefficients `[ΔC20, ΔC21, ΔS21, ΔC22, ΔS22]`, from the Greenwich mean sidereal time `gmst` and the
/// five Delaunay arguments `[l, l', F, D, Ω]` (all radians). This is the ~mm frequency dependence that
/// the frequency-independent Step-1 Love numbers omit, matching Orekit's tide-frequency series.
#[pyfunction]
fn solid_tide_step2(gmst: f64, delaunay: [f64; 5]) -> [f64; 5] {
    spacelab_propagation::step2_corrections(gmst, delaunay)
}

/// Which conventional pole the pole-tide wobble variables are measured from: `"secular2018"` (the
/// default, Chapter 7 Eq. 21 as amended 2018/02/01) or `"iers2010"` (the retired Table 7.7 mean
/// pole, which is what Orekit implements).
fn mean_pole(name: &str) -> PyResult<spacelab_propagation::MeanPole> {
    match name {
        "secular2018" => Ok(spacelab_propagation::MeanPole::Secular2018),
        "iers2010" => Ok(spacelab_propagation::MeanPole::Iers2010),
        _ => Err(PyValueError::new_err(
            "mean_pole must be 'secular2018' or 'iers2010'",
        )),
    }
}

/// IERS solid Earth pole tide (§6.4, Eq. 6.23): the degree-2 order-1 corrections `[ΔC21, ΔS21]` from
/// the polar motion `(xp, yp)` (radians) about the conventional pole `mean_pole` names, with `t` in
/// seconds since J2000 (TT). Add these to the ΔC21/ΔS21 of `solid_tide_step2` for the full degree-2
/// tide. The two poles are 111 mas apart by 2024, enough to flip the sign of ΔC21, so `"iers2010"`
/// is for reproducing an Orekit result and not for a current computation.
#[pyfunction]
#[pyo3(signature = (xp, yp, t, mean_pole = "secular2018"))]
fn solid_pole_tide(xp: f64, yp: f64, t: f64, mean_pole: &str) -> PyResult<[f64; 2]> {
    let pole = self::mean_pole(mean_pole)?;
    Ok(spacelab_propagation::pole_tide_corrections(xp, yp, t, pole))
}

/// IERS ocean pole tide (§6.5, Eq. 6.24): the degree-2 order-1 corrections `[ΔC21, ΔS21]` from the
/// polar motion `(xp, yp)` (radians), `t` seconds since J2000 (TT). Superposes with the solid pole
/// tide on the ΔC21/ΔS21 band and must be given the same `mean_pole`, since §6.4 and §6.5 share the
/// definition of the wobble variables.
#[pyfunction]
#[pyo3(signature = (xp, yp, t, mean_pole = "secular2018"))]
fn ocean_pole_tide(xp: f64, yp: f64, t: f64, mean_pole: &str) -> PyResult<[f64; 2]> {
    let pole = self::mean_pole(mean_pole)?;
    Ok(spacelab_propagation::ocean_pole_tide_corrections(
        xp, yp, t, pole,
    ))
}

/// Combined DSST **mean** propagation: the zonal geopotential (J2..Jn), Sun + Moon third-body
/// attraction and solar radiation pressure summed into one mean-element right-hand side and
/// integrated with DOP853. Given an initial *mean* equinoctial state `[a, k, h, q, p, λ]`, returns
/// the mean equinoctial state at each requested time.
///
/// The Sun and Moon geocentric positions come from the built-in analytic ephemerides (advanced along
/// the J2000 timeline by `epoch_j2000_seconds`). The resonant tesseral contribution is omitted: its
/// mean rate is identically zero away from a tesseral resonance, i.e. for every orbit except
/// repeat-ground-track ones (GEO, GPS), so this is the full conservative mean rate for the common
/// LEO/MEO/HEO case. Each summed contribution is individually bit-exact against Orekit.
#[pyfunction]
#[pyo3(signature = (
    initial, times, epoch_j2000_seconds,
    mu = 3.986_004_415e14, re = 6_378_136.3,
    j = vec![1.082_626_68e-3, -2.532_656_49e-6, -1.619_621_59e-6, -2.272_960_83e-7],
    cr = 1.5, area_over_mass = 0.02,
))]
#[allow(clippy::too_many_arguments)]
fn dsst_propagate_mean_full(
    py: Python<'_>,
    initial: [f64; 6],
    times: Vec<f64>,
    epoch_j2000_seconds: f64,
    mu: f64,
    re: f64,
    j: Vec<f64>,
    cr: f64,
    area_over_mass: f64,
) -> Vec<[f64; 6]> {
    const MU_SUN: f64 = 1.327_124_400_18e20;
    const MU_MOON: f64 = 4.902_800_066e12;
    const EARTH_ROTATION_RATE: f64 = 7.292_115_146_706_979e-5;
    let zonal = DsstZonal::from_j(mu, re, &j);
    let sun_tb = DsstThirdBody::new(mu, MU_SUN);
    let moon_tb = DsstThirdBody::new(mu, MU_MOON);
    let srp = DsstSolarRadiationPressure::new(cr, area_over_mass, mu, re);
    // Zero tesseral field: the tesseral MEAN rate is nonzero only at a resonance.
    let tesseral = DsstTesseral::new(
        mu,
        re,
        EARTH_ROTATION_RATE,
        std::vec![std::vec![0.0; 3]; 3],
        std::vec![std::vec![0.0; 3]; 3],
    );
    let sun = AnalyticSun {
        epoch_j2000_seconds,
    };
    let moon = AnalyticMoon {
        epoch_j2000_seconds,
    };
    let prop = DsstMeanPropagator {
        zonal: &zonal,
        sun: &sun_tb,
        moon: &moon_tb,
        srp: &srp,
        tesseral: &tesseral,
        ephemeris: |t: f64| (sun.position(t), moon.position(t)),
        gmst: |_t: f64| 0.0, // unused: the tesseral field is zero
        mu,
    };
    py.detach(|| times.iter().map(|&t| prop.propagate(initial, t)).collect())
}

/// Combined DSST **osculating** propagation: the mean propagation of [`dsst_propagate_mean_full`]
/// plus the sum of each contribution's short-period correction (zonal + Sun + Moon + SRP), so the
/// returned equinoctial states are osculating rather than mean. Given an initial *mean* equinoctial
/// state `[a, k, h, q, p, λ]`, returns the osculating state at each requested time. The tesseral
/// short-period is zero here (non-resonant, zero field); drag is not part of this propagator.
#[pyfunction]
#[pyo3(signature = (
    initial, times, epoch_j2000_seconds,
    mu = 3.986_004_415e14, re = 6_378_136.3,
    j = vec![1.082_626_68e-3, -2.532_656_49e-6, -1.619_621_59e-6, -2.272_960_83e-7],
    cr = 1.5, area_over_mass = 0.02,
))]
#[allow(clippy::too_many_arguments)]
fn dsst_propagate_osculating_full(
    py: Python<'_>,
    initial: [f64; 6],
    times: Vec<f64>,
    epoch_j2000_seconds: f64,
    mu: f64,
    re: f64,
    j: Vec<f64>,
    cr: f64,
    area_over_mass: f64,
) -> Vec<[f64; 6]> {
    const MU_SUN: f64 = 1.327_124_400_18e20;
    const MU_MOON: f64 = 4.902_800_066e12;
    const EARTH_ROTATION_RATE: f64 = 7.292_115_146_706_979e-5;
    let zonal = DsstZonal::from_j(mu, re, &j);
    let sun_tb = DsstThirdBody::new(mu, MU_SUN);
    let moon_tb = DsstThirdBody::new(mu, MU_MOON);
    let srp = DsstSolarRadiationPressure::new(cr, area_over_mass, mu, re);
    let tesseral = DsstTesseral::new(
        mu,
        re,
        EARTH_ROTATION_RATE,
        std::vec![std::vec![0.0; 3]; 3],
        std::vec![std::vec![0.0; 3]; 3],
    );
    let sun = AnalyticSun {
        epoch_j2000_seconds,
    };
    let moon = AnalyticMoon {
        epoch_j2000_seconds,
    };
    let prop = DsstMeanPropagator {
        zonal: &zonal,
        sun: &sun_tb,
        moon: &moon_tb,
        srp: &srp,
        tesseral: &tesseral,
        ephemeris: |t: f64| (sun.position(t), moon.position(t)),
        gmst: |_t: f64| 0.0,
        mu,
    };
    py.detach(|| {
        times
            .iter()
            .map(|&t| prop.propagate_osculating(initial, t))
            .collect()
    })
}

/// Walker constellation `i: T/P/F` — returns each satellite's inertial state `(r, v)` (SI) at the
/// epoch, plane-major. `pattern` is `"delta"` (planes over 2π of RAAN) or `"star"` (over π); angles
/// are radians, `semi_major_axis` metres. Feed the states to any propagator or plot them directly.
#[pyfunction]
#[pyo3(signature = (pattern, inclination, planes, per_plane, phasing, semi_major_axis, mu = 3.986_004_418e14))]
fn walker_constellation(
    pattern: &str,
    inclination: f64,
    planes: u32,
    per_plane: u32,
    phasing: u32,
    semi_major_axis: f64,
    mu: f64,
) -> PyResult<Vec<([f64; 3], [f64; 3])>> {
    let pattern = match pattern {
        "delta" => WalkerPattern::Delta,
        "star" => WalkerPattern::Star,
        _ => return Err(PyValueError::new_err("pattern must be 'delta' or 'star'")),
    };
    let w = Walker {
        pattern,
        inclination: Angle::from_radians(inclination),
        planes,
        per_plane,
        phasing,
        semi_major_axis,
        mu,
    };
    let mut out = Vec::new();
    w.generate(|e| {
        let (r, v) = e.to_state();
        out.push((arr(r), arr(v)));
    });
    Ok(out)
}

/// Near-polar Street-of-Coverage constellation sized for continuous single-fold coverage above
/// `min_elevation` (rad) from a circular orbit at `altitude` (m). Returns the satellites' inertial
/// states `(r, v)` (SI), plane-major; the list is **empty** when the geometry cannot tile a street
/// (footprint ≤ 0, an inclination too far from polar, or a `per_plane` below the `⌈π/θ⌉` floor).
/// `per_plane = None` picks the minimum. Use [`street_of_coverage_geometry`] for the plane count.
#[pyfunction]
#[pyo3(signature = (inclination, altitude, min_elevation, earth_radius = 6_378_137.0, mu = 3.986_004_418e14, per_plane = None))]
fn street_of_coverage(
    inclination: f64,
    altitude: f64,
    min_elevation: f64,
    earth_radius: f64,
    mu: f64,
    per_plane: Option<u32>,
) -> Vec<([f64; 3], [f64; 3])> {
    let soc = StreetOfCoverage {
        inclination: Angle::from_radians(inclination),
        altitude,
        min_elevation: Angle::from_radians(min_elevation),
        earth_radius,
        mu,
        per_plane,
    };
    let mut out = Vec::new();
    soc.generate(|e| {
        let (r, v) = e.to_state();
        out.push((arr(r), arr(v)));
    });
    out
}

/// The solved Street-of-Coverage geometry for the same inputs as [`street_of_coverage`], as
/// `(planes, per_plane, footprint_half_angle, street_half_width, raan_step_co, raan_step_counter,
/// raan_step, raan_span, inter_plane_phase)` — counts as integers, angles in radians. Returns `None`
/// when the geometry is infeasible.
#[pyfunction]
#[pyo3(signature = (inclination, altitude, min_elevation, earth_radius = 6_378_137.0, mu = 3.986_004_418e14, per_plane = None))]
#[allow(clippy::type_complexity)] // one flat geometry record, documented field-by-field above
fn street_of_coverage_geometry(
    inclination: f64,
    altitude: f64,
    min_elevation: f64,
    earth_radius: f64,
    mu: f64,
    per_plane: Option<u32>,
) -> Option<(u32, u32, f64, f64, f64, f64, f64, f64, f64)> {
    let soc = StreetOfCoverage {
        inclination: Angle::from_radians(inclination),
        altitude,
        min_elevation: Angle::from_radians(min_elevation),
        earth_radius,
        mu,
        per_plane,
    };
    soc.geometry().map(|g| {
        (
            g.planes,
            g.per_plane,
            g.footprint_half_angle,
            g.street_half_width,
            g.raan_step_co,
            g.raan_step_counter,
            g.raan_step,
            g.raan_span,
            g.inter_plane_phase,
        )
    })
}

/// Flower constellation — `satellites` spacecraft on one `(petals, days)` repeating orbit of shape
/// `(eccentricity, inclination, arg_periapsis)`, uniformly time-phased along the common ground track
/// (Mortari et al. 2004). `frame_rate` is the rotating frame's rate (rad/s, default Earth sidereal);
/// angles radians. Returns the inertial states `(r, v)` (SI), or empty if any count is zero.
#[pyfunction]
#[pyo3(signature = (
    petals, days, satellites, eccentricity, inclination, arg_periapsis,
    raan = 0.0, mean_anomaly = 0.0, frame_rate = 7.292_115_146_5e-5, mu = 3.986_004_418e14,
))]
#[allow(clippy::too_many_arguments)] // Np, Nd, Ns, the shared orbit shape, the reference Ω/M, frame, mu
fn flower_constellation(
    petals: u32,
    days: u32,
    satellites: u32,
    eccentricity: f64,
    inclination: f64,
    arg_periapsis: f64,
    raan: f64,
    mean_anomaly: f64,
    frame_rate: f64,
    mu: f64,
) -> Vec<([f64; 3], [f64; 3])> {
    let fc = FlowerConstellation {
        petals,
        days,
        satellites,
        eccentricity,
        inclination: Angle::from_radians(inclination),
        arg_periapsis: Angle::from_radians(arg_periapsis),
        raan: Angle::from_radians(raan),
        mean_anomaly: Angle::from_radians(mean_anomaly),
        frame_rate,
        mu,
    };
    let mut out = Vec::new();
    fc.generate(|e| {
        let (r, v) = e.to_state();
        out.push((arr(r), arr(v)));
    });
    out
}

/// 2-D Lattice Flower Constellation — the integer-lattice unification of Walker and Flower. `orbits`
/// right-ascension slots × `sats_per_orbit` per slot, placed by configuration number `config`; every
/// satellite shares the orbit `(semi_major_axis, eccentricity, inclination, arg_periapsis)`. Returns
/// the inertial states `(r, v)` (SI). A circular lattice with `config = F` reproduces a Walker
/// `i: T/P/F`; the literature's `Nc′` (Avendaño et al. 2013) is `config = −Nc′ mod orbits`
/// (`docs/adr/0002-lattice-configuration-number.md`). Angles radians.
#[pyfunction]
#[pyo3(signature = (
    orbits, sats_per_orbit, config, semi_major_axis, eccentricity, inclination,
    arg_periapsis = 0.0, raan0 = 0.0, mu = 3.986_004_418e14,
))]
#[allow(clippy::too_many_arguments)] // lattice (No, Nso, Nc) + the shared orbit shape + base RAAN + mu
fn lattice_flower(
    orbits: u32,
    sats_per_orbit: u32,
    config: u32,
    semi_major_axis: f64,
    eccentricity: f64,
    inclination: f64,
    arg_periapsis: f64,
    raan0: f64,
    mu: f64,
) -> Vec<([f64; 3], [f64; 3])> {
    let lattice = LatticeFlower {
        orbits,
        sats_per_orbit,
        config,
        semi_major_axis,
        eccentricity,
        inclination: Angle::from_radians(inclination),
        arg_periapsis: Angle::from_radians(arg_periapsis),
        raan0: Angle::from_radians(raan0),
        mu,
    };
    let mut out = Vec::new();
    lattice.generate(|e| {
        let (r, v) = e.to_state();
        out.push((arr(r), arr(v)));
    });
    out
}

/// Draim's four-satellite continuous global coverage constellation (Draim 1987). `period` seconds,
/// angles radians; the defaults are the patent's worked example (48 h, `e = 0.28`, `i = 33°`).
/// Returns the four satellites' inertial states `(r, v)` (SI).
#[pyfunction]
#[pyo3(signature = (period = 172_800.0, eccentricity = 0.28, inclination = 0.575_958_653_158_128_8, raan0 = 0.0, mu = 3.986_004_418e14))]
fn draim_constellation(
    period: f64,
    eccentricity: f64,
    inclination: f64,
    raan0: f64,
    mu: f64,
) -> Vec<([f64; 3], [f64; 3])> {
    let draim = DraimTetrahedral::from_period(
        period,
        eccentricity,
        Angle::from_radians(inclination),
        Angle::from_radians(raan0),
        mu,
    );
    let mut out = Vec::new();
    draim.generate(|e| {
        let (r, v) = e.to_state();
        out.push((arr(r), arr(v)));
    });
    out
}

/// The α-characteristic (radians) of a delta-system given its satellites' inertial states `(r, v)` at
/// epoch and `mu`: the smallest sub-satellite footprint at which the constellation continuously
/// `fold`-fold covers the sphere. Evaluated on a `grid_points` × `time_samples` space-time grid.
#[pyfunction]
#[pyo3(signature = (states, mu, fold = 1, grid_points = 500, time_samples = 60))]
fn alpha_characteristic(
    states: Vec<([f64; 3], [f64; 3])>,
    mu: f64,
    fold: u32,
    grid_points: u32,
    time_samples: u32,
) -> f64 {
    let sats: Vec<KeplerianElements> = states
        .iter()
        .map(|(r, v)| KeplerianElements::from_state(vec3(*r), vec3(*v), mu))
        .collect();
    spacelab_constellation::alpha_characteristic(&sats, fold, grid_points, time_samples)
}

/// The fewest-satellite Walker Delta system whose α-characteristic is within `max_footprint`
/// (radians) — the cheapest constellation that continuously `fold`-fold covers the sphere with that
/// sensor footprint, at the common orbit. Returns `(planes, per_plane, phasing, total)`, or `None`.
#[pyfunction]
#[pyo3(signature = (
    max_footprint, inclination, semi_major_axis, mu = 3.986_004_418e14,
    fold = 1, max_total = 24, grid_points = 400, time_samples = 40,
))]
#[allow(clippy::too_many_arguments)]
fn minimum_walker(
    max_footprint: f64,
    inclination: f64,
    semi_major_axis: f64,
    mu: f64,
    fold: u32,
    max_total: u32,
    grid_points: u32,
    time_samples: u32,
) -> Option<(u32, u32, u32, u32)> {
    spacelab_constellation::minimum_walker(
        max_footprint,
        Angle::from_radians(inclination),
        semi_major_axis,
        mu,
        fold,
        max_total,
        grid_points,
        time_samples,
    )
    .map(|w| (w.planes, w.per_plane, w.phasing, w.total()))
}

/// The Draim orbit geometry for the given `period` (s), eccentricity and inclination: returns
/// `(semi_major_axis, perigee_radius, apogee_radius)` in metres.
#[pyfunction]
#[pyo3(signature = (period = 172_800.0, eccentricity = 0.28, inclination = 0.575_958_653_158_128_8, mu = 3.986_004_418e14))]
fn draim_geometry(period: f64, eccentricity: f64, inclination: f64, mu: f64) -> (f64, f64, f64) {
    let d = DraimTetrahedral::from_period(
        period,
        eccentricity,
        Angle::from_radians(inclination),
        Angle::from_radians(0.0),
        mu,
    );
    let a = d.semi_major_axis;
    (a, a * (1.0 - eccentricity), a * (1.0 + eccentricity))
}

/// Propagate every GPS/Galileo/BeiDou/QZSS/NavIC record in a RINEX-3 navigation `text` to `tk` seconds
/// from each record's reference epoch, returning `[(system, prn, [x, y, z], [vx, vy, vz])]` in ECEF
/// (m, m/s). `system` is the RINEX-3 constellation letter — `G`, `E`, `C`, `J` or `I` — without which
/// a PRN is ambiguous across constellations in a mixed file.
#[pyfunction]
fn rinex_gnss_positions(text: &str, tk: f64) -> Vec<(char, u8, [f64; 3], [f64; 3])> {
    parse_nav(text)
        .into_iter()
        .map(|e| {
            let (p, v) = e.position_velocity_ecef(tk);
            let system = match e.system {
                GnssSystem::Gps => 'G',
                GnssSystem::Galileo => 'E',
                GnssSystem::Beidou => 'C',
                GnssSystem::Qzss => 'J',
                GnssSystem::Navic => 'I',
            };
            (system, e.prn, p, v)
        })
        .collect()
}

/// Propagate every GLONASS record in a RINEX-3 navigation `text` to `dt` seconds from each record's
/// reference epoch, returning `[(slot, [x, y, z], [vx, vy, vz])]` in PZ-90 (m, m/s).
#[pyfunction]
fn rinex_glonass_positions(text: &str, dt: f64) -> Vec<(u8, [f64; 3], [f64; 3])> {
    parse_glonass_nav(text)
        .into_iter()
        .map(|g| {
            let (p, v) = g.position_velocity_pz90(dt);
            (g.slot, p, v)
        })
        .collect()
}

/// Dilution of precision for a `receiver` (ECEF, m) seeing `satellites` (ECEF, m): returns
/// `(gdop, pdop, hdop, vdop, tdop)`, or `None` with fewer than four satellites or a singular geometry.
#[pyfunction]
fn dilution_of_precision(
    receiver: [f64; 3],
    satellites: Vec<[f64; 3]>,
) -> Option<(f64, f64, f64, f64, f64)> {
    gnss_dop(receiver, &satellites).map(|d| (d.gdop, d.pdop, d.hdop, d.vdop, d.tdop))
}

/// The broadcast (Klobuchar) L1 ionospheric slant delay in metres, from the eight broadcast
/// coefficients, the receiver geodetic `lat`/`lon` (rad), the satellite `elevation`/`azimuth` (rad) and
/// GPS time `gps_sod` (seconds of day).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn klobuchar_iono_delay(
    alpha: [f64; 4],
    beta: [f64; 4],
    lat: f64,
    lon: f64,
    elevation: f64,
    azimuth: f64,
    gps_sod: f64,
) -> f64 {
    gnss_klobuchar(alpha, beta, lat, lon, elevation, azimuth, gps_sod)
}

/// The Saastamoinen zenith hydrostatic tropospheric delay in metres, from surface `pressure` (hPa),
/// geodetic `lat` (rad) and ellipsoidal `height` (m).
#[pyfunction]
fn saastamoinen_zenith_delay(pressure: f64, lat: f64, height: f64) -> f64 {
    gnss_saastamoinen(pressure, lat, height)
}

/// The Sun-synchronous inclination (radians) for a circular-ish orbit of `semi_major_axis` (m) and
/// `eccentricity`. `None` when the orbit is too high for `J2` to precess the node a full turn a year.
/// Defaults are Earth (WGS-84 μ/Rₑ/J₂) and the tropical-year node rate.
#[pyfunction]
#[pyo3(signature = (semi_major_axis, eccentricity=0.0, mu=3.986_004_418e14, re=6_378_137.0, j2=1.082_626_68e-3))]
fn sun_synchronous_inclination(
    semi_major_axis: f64,
    eccentricity: f64,
    mu: f64,
    re: f64,
    j2: f64,
) -> Option<f64> {
    design_sun_synchronous_inclination(
        semi_major_axis,
        eccentricity,
        mu,
        re,
        j2,
        SUN_SYNCHRONOUS_NODE_RATE,
    )
    .map(Angle::radians)
}

/// The semi-major axis (m) whose ground track repeats after `revs` revolutions in `days` nodal days at
/// the given `inclination` (rad). Defaults are Earth (WGS-84 μ/Rₑ/J₂ and sidereal rotation rate).
#[pyfunction]
#[pyo3(signature = (revs, days, inclination, eccentricity=0.0, mu=3.986_004_418e14, re=6_378_137.0, j2=1.082_626_68e-3, earth_rotation_rate=7.292_115_146_7e-5))]
#[allow(clippy::too_many_arguments)]
fn repeat_ground_track_sma(
    revs: u32,
    days: u32,
    inclination: f64,
    eccentricity: f64,
    mu: f64,
    re: f64,
    j2: f64,
    earth_rotation_rate: f64,
) -> f64 {
    design_repeat_ground_track_sma(
        revs,
        days,
        eccentricity,
        Angle::from_radians(inclination),
        mu,
        re,
        j2,
        earth_rotation_rate,
    )
}

/// The frozen-orbit eccentricity (`ω = 90°`) balancing the `J2`/`J3` perigee drift, for
/// `semi_major_axis` (m) and `inclination` (rad). Defaults are Earth (WGS-84 Rₑ, EGM `J2`/`J3`).
#[pyfunction]
#[pyo3(signature = (semi_major_axis, inclination, re=6_378_137.0, j2=1.082_626_68e-3, j3=-2.532_653_65e-6))]
fn frozen_eccentricity(semi_major_axis: f64, inclination: f64, re: f64, j2: f64, j3: f64) -> f64 {
    design_frozen_eccentricity(
        semi_major_axis,
        Angle::from_radians(inclination),
        re,
        j2,
        j3,
    )
}

/// The prograde critical inclination (radians), `acos(1/√5) ≈ 63.43°`, where apsidal precession
/// vanishes.
#[pyfunction]
fn critical_inclination() -> f64 {
    design_critical_inclination().radians()
}

/// The NeQuick-G (Galileo) slant total electron content (TECU) along the receiver→satellite ray.
/// `alpha` are the three broadcast effective-ionisation coefficients; `month` (1–12) and
/// `seconds_in_utc_day` set the epoch; receiver/satellite are geodetic (latitude, longitude in radians,
/// altitude in metres).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn nequick_galileo_stec(
    alpha: [f64; 3],
    month: u32,
    seconds_in_utc_day: f64,
    receiver_lat: f64,
    receiver_lon: f64,
    receiver_alt: f64,
    satellite_lat: f64,
    satellite_lon: f64,
    satellite_alt: f64,
) -> f64 {
    NeQuickGalileo::new(alpha).stec(
        DateTimeComponents::new(month, seconds_in_utc_day),
        GeodeticPoint::new(receiver_lat, receiver_lon, receiver_alt),
        GeodeticPoint::new(satellite_lat, satellite_lon, satellite_alt),
    )
}

/// True → apparent altitude (degrees), Sæmundsson (1986) — the angle a telescope points at, given the
/// one an ephemeris computes. `None` outside the fit's domain, which closes 1° below the horizon.
#[pyfunction]
#[pyo3(signature = (true_altitude_deg, pressure_hpa = 1010.0, temperature_k = 283.0))]
fn refraction_apparent(
    true_altitude_deg: f64,
    pressure_hpa: f64,
    temperature_k: f64,
) -> Option<f64> {
    Refraction {
        pressure_hpa,
        temperature_k,
    }
    .apparent(TrueAltitude(true_altitude_deg.to_radians()))
    .map(|h| h.0.to_degrees())
}

/// Apparent → true altitude (degrees), Bennett (1982) — the conversion [`Sensor.elevation_apparent`]
/// applies to a mask. `None` below the horizon, where there is no ray to have been bent.
///
/// The two fits are not each other's inverse: a round trip closes to a few arcseconds near the
/// horizon, which is the published disagreement between them and not an error in either.
#[pyfunction]
#[pyo3(signature = (apparent_altitude_deg, pressure_hpa = 1010.0, temperature_k = 283.0))]
fn refraction_geometric(
    apparent_altitude_deg: f64,
    pressure_hpa: f64,
    temperature_k: f64,
) -> Option<f64> {
    Refraction {
        pressure_hpa,
        temperature_k,
    }
    .geometric(ApparentAltitude(apparent_altitude_deg.to_radians()))
    .map(|h| h.0.to_degrees())
}

/// The velocity a station is carried at by the rotation of the body it stands on, `ω × r` (m/s),
/// from its position (m) and the rotation vector (rad/s) in the same frame.
///
/// For a ground station observing in TEME that is the station's TEME position and
/// `(0, 0, 7.292115146706979e-5)`. The rotation is a *vector* rather than a scalar rate so that
/// polar motion can tilt the pole.
#[pyfunction]
fn diurnal_velocity(station_m: [f64; 3], angular_velocity_rad_s: [f64; 3]) -> [f64; 3] {
    arr(vec3(angular_velocity_rad_s).cross(vec3(station_m)))
}

/// Natural → apparent direction: where an observer moving at `velocity_mps` sees a source that is
/// really in the direction `natural`. Unit vector out, whatever the input's length.
///
/// `None` for a velocity at or above the speed of light or a direction of zero length.
///
/// # Which velocity
///
/// Not the Earth's 29.8 km/s orbital motion — the famous 20.5″ constant of aberration. A station and
/// the satellite it is watching share that velocity, and a shared velocity cancels in a geocentric
/// frame. What is left is the station's own rotation, [`diurnal_velocity`]: 465.1 m/s at the equator,
/// falling as `cos φ`, worth **0.320″** at the equator and **0.262″** at 34.9°. That is a hundred
/// times an astrometric plate's centroiding error, and it turns with the hour angle rather than
/// averaging out.
#[pyfunction]
fn aberration_apply(velocity_mps: [f64; 3], natural: [f64; 3]) -> Option<[f64; 3]> {
    Aberration::new(vec3(velocity_mps))?
        .apply(vec3(natural))
        .map(arr)
}

/// Apparent → natural direction: the correction applied to a measurement, the exact inverse of
/// [`aberration_apply`] rather than a sign flip on a small-angle term.
///
/// `None` on the same two conditions as [`aberration_apply`].
#[pyfunction]
fn aberration_remove(velocity_mps: [f64; 3], apparent: [f64; 3]) -> Option<[f64; 3]> {
    Aberration::new(vec3(velocity_mps))?
        .remove(vec3(apparent))
        .map(arr)
}

/// The first-order ionospheric range delay (metres), `Δ = 40.3·TEC/f²`, from a slant TEC (TECU) and
/// the carrier frequency (Hz).
#[pyfunction]
fn ionospheric_path_delay(stec_tecu: f64, frequency_hz: f64) -> f64 {
    iono_path_delay(stec_tecu, frequency_hz)
}

/// The second-order ionospheric group delay (metres) — the geomagnetic `1/f³` term Orekit omits, which
/// survives an ionosphere-free combination as an L1/L2 asymmetry. From receiver geocentric `latitude`/
/// `longitude` (rad), signal `elevation`/`azimuth` (rad), slant TEC (TECU) and carrier `frequency` (Hz),
/// with the ionospheric shell at `pierce_height_m` (default 350 km). Signed. Uses the IGRF-14 (2025.0)
/// centred dipole for `B·cos θ` at the pierce point (Bassiri–Hajj / Navipedia).
#[pyfunction]
#[pyo3(signature = (latitude, longitude, elevation, azimuth, stec_tecu, frequency_hz, pierce_height_m = 350_000.0))]
fn second_order_ionospheric_delay(
    latitude: f64,
    longitude: f64,
    elevation: f64,
    azimuth: f64,
    stec_tecu: f64,
    frequency_hz: f64,
    pierce_height_m: f64,
) -> f64 {
    gnss_hoi2(
        latitude,
        longitude,
        elevation,
        azimuth,
        pierce_height_m,
        stec_tecu * 1e16,
        frequency_hz,
    )
}

/// Earth's magnetic field vector `(Bx, By, Bz)` (Tesla, ECEF) at ECEF position `(x, y, z)` (m) and
/// decimal `year`, from the full IGRF-14 model (degree/order 13). Valid ~2025-2030.
#[pyfunction]
fn igrf_field_ecef(x: f64, y: f64, z: f64, year: f64) -> (f64, f64, f64) {
    let b = gnss_igrf_field([x, y, z], year);
    (b[0], b[1], b[2])
}

/// The Niell (1996) tropospheric mapping functions `(hydrostatic, wet)` for geodetic `latitude` (rad),
/// ellipsoidal `height` (m), `day_of_year` and `elevation` (rad).
#[pyfunction]
fn niell_mapping(latitude: f64, height: f64, day_of_year: f64, elevation: f64) -> (f64, f64) {
    gnss_niell_mapping(latitude, height, day_of_year, elevation)
}

/// The Global Mapping Function (GMF, Böhm 2006) tropospheric mapping functions `(hydrostatic, wet)`
/// for geodetic `latitude`/`longitude` (rad), ellipsoidal `height` (m), `day_of_year` and `elevation`
/// (rad).
#[pyfunction]
fn gmf_mapping(
    latitude: f64,
    longitude: f64,
    height: f64,
    day_of_year: f64,
    elevation: f64,
) -> (f64, f64) {
    gnss_gmf_mapping(latitude, longitude, height, day_of_year, elevation)
}

/// The VMF1 (Vienna, Böhm et al. 2006) tropospheric mapping functions `(hydrostatic, wet)`. The
/// `a_h`/`a_w` coefficients come from a numerical weather model; geodetic `latitude` (rad),
/// ellipsoidal `height` (m), `day_of_year` and `elevation` (rad) place the site and ray.
#[pyfunction]
fn vmf1_mapping(
    a_h: f64,
    a_w: f64,
    latitude: f64,
    height: f64,
    day_of_year: f64,
    elevation: f64,
) -> (f64, f64) {
    gnss_vmf1_mapping(a_h, a_w, latitude, height, day_of_year, elevation)
}

/// The VMF3 (Vienna, Landskron & Böhm 2018) tropospheric mapping functions `(hydrostatic, wet)`. The
/// `a_h`/`a_w` coefficients come from a numerical weather model; geodetic `latitude`/`longitude`
/// (rad), ellipsoidal `height` (m), `day_of_year` and `elevation` (rad) place the site and ray.
#[pyfunction]
fn vmf3_mapping(
    a_h: f64,
    a_w: f64,
    latitude: f64,
    longitude: f64,
    height: f64,
    day_of_year: f64,
    elevation: f64,
) -> (f64, f64) {
    gnss_vmf3_mapping(
        a_h,
        a_w,
        latitude,
        longitude,
        height,
        day_of_year,
        elevation,
    )
}

/// Bilinearly interpolate an operational VMF grid file (its full text) at geodetic `latitude`/
/// `longitude` (rad), returning `(ah, aw, zhd, zwd)`. Feed `ah`/`aw` to `vmf1_mapping`/`vmf3_mapping`.
#[pyfunction]
fn vmf_grid_coefficients(
    grid_text: &str,
    latitude: f64,
    longitude: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    let grid = VmfGrid::parse(grid_text).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let c = grid.interpolate(latitude, longitude);
    Ok((c.ah, c.aw, c.zhd, c.zwd))
}

/// GPT3 (Landskron & Böhm 2018) empirical/blind troposphere at geodetic `latitude`/`longitude` (rad),
/// fractional `day_of_year` and ellipsoidal `height` (m). Returns a dict with the VMF3 mapping
/// coefficients `ah`/`aw` plus `pressure` (Pa), `temperature` (K), `water_vapor_pressure` (Pa), `tm`
/// (K), `lambda`, and the gradients `gnh`/`geh`/`gnw`/`gew`. Feed `ah`/`aw` to `vmf3_mapping`.
#[pyfunction]
fn gpt3(
    latitude: f64,
    longitude: f64,
    day_of_year: f64,
    height: f64,
) -> std::collections::HashMap<&'static str, f64> {
    let r = gnss_gpt3(latitude, longitude, day_of_year, height);
    std::collections::HashMap::from([
        ("ah", r.ah),
        ("aw", r.aw),
        ("pressure", r.pressure),
        ("temperature", r.temperature),
        ("water_vapor_pressure", r.water_vapor_pressure),
        ("tm", r.tm),
        ("lambda", r.lambda),
        ("gnh", r.gnh),
        ("geh", r.geh),
        ("gnw", r.gnw),
        ("gew", r.gew),
    ])
}

/// Fit an open, HASDM-style thermospheric density correction (degree 4, 25 spherical-harmonic
/// coefficients) by weighted least squares to samples of `ln(rho_observed / rho_base)` at sun-fixed
/// `latitudes` (rad) and `local_solar_times` (hours), with per-sample `weights`. Returns the 25 packed
/// coefficients, usable with `density_correction_factor`. A small positive `ridge` (Tikhonov)
/// stabilizes the coefficients when the samples span little local-solar-time (e.g. one satellite-month).
#[pyfunction]
#[pyo3(signature = (latitudes, local_solar_times, log_ratios, weights, ridge = 0.0))]
fn density_correction_fit(
    latitudes: Vec<f64>,
    local_solar_times: Vec<f64>,
    log_ratios: Vec<f64>,
    weights: Vec<f64>,
    ridge: f64,
) -> PyResult<Vec<f64>> {
    let n = latitudes.len();
    if local_solar_times.len() != n || log_ratios.len() != n || weights.len() != n {
        return Err(PyValueError::new_err(
            "density_correction_fit: latitudes, local_solar_times, log_ratios and weights must be the same length",
        ));
    }
    let samples: Vec<DensitySample> = (0..n)
        .map(|i| DensitySample {
            latitude: latitudes[i],
            local_solar_time: local_solar_times[i],
            log_ratio: log_ratios[i],
            weight: weights[i],
        })
        .collect();
    let correction = DensityCorrection::<25>::fit_ridge(&samples, ridge).ok_or_else(|| {
        PyValueError::new_err(
            "density_correction_fit: singular or underdetermined normal equations",
        )
    })?;
    Ok(correction.coefficients().to_vec())
}

/// The multiplicative density correction factor `rho_corrected / rho_base` at sun-fixed `latitude`
/// (rad) and `local_solar_time` (hours), from the 25 packed degree-4 coefficients.
#[pyfunction]
fn density_correction_factor(
    coefficients: Vec<f64>,
    latitude: f64,
    local_solar_time: f64,
) -> PyResult<f64> {
    let coeffs: [f64; 25] = coefficients.try_into().map_err(|_| {
        PyValueError::new_err("density_correction_factor: expected 25 coefficients (degree 4)")
    })?;
    Ok(DensityCorrection::<25>::from_coefficients(coeffs).factor(latitude, local_solar_time))
}

/// Parse a TU Delft thermosphere density file (CHAMP/GRACE/GRACE-FO/Swarm, Siemes et al. 2023, CC BY
/// 4.0) and return calibration samples `(latitudes_rad, local_solar_times_h, log_ratios, weights)` of
/// `ln(rho / rho_orbit_mean)`, ready for `density_correction_fit`.
#[pyfunction]
fn density_data_samples(text: &str) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let samples = orbit_mean_samples(&parse_density(text));
    (
        samples.iter().map(|s| s.latitude).collect(),
        samples.iter().map(|s| s.local_solar_time).collect(),
        samples.iter().map(|s| s.log_ratio).collect(),
        samples.iter().map(|s| s.weight).collect(),
    )
}

/// Bin density-ratio samples into an `n_lat x n_lst` field of the mean `log_ratio` (row-major
/// `lat_bin*n_lst + lst_bin`, empty cells 0.0) — a density-field snapshot for `DensityRom` POD.
#[pyfunction]
fn density_grid_snapshot(
    latitudes: Vec<f64>,
    local_solar_times: Vec<f64>,
    log_ratios: Vec<f64>,
    n_lat: usize,
    n_lst: usize,
) -> PyResult<Vec<f64>> {
    let n = latitudes.len();
    if local_solar_times.len() != n || log_ratios.len() != n {
        return Err(PyValueError::new_err(
            "density_grid_snapshot: latitudes, local_solar_times and log_ratios must be the same length",
        ));
    }
    let samples: Vec<DensitySample> = (0..n)
        .map(|i| DensitySample {
            latitude: latitudes[i],
            local_solar_time: local_solar_times[i],
            log_ratio: log_ratios[i],
            weight: 1.0,
        })
        .collect();
    Ok(grid_samples(&samples, n_lat, n_lst))
}

/// Parse a CelesTrak `SW-All.csv` file into daily space-weather arrays `(years, months, days, ap,
/// f107_observed, f107_center81)`.
#[allow(clippy::type_complexity)] // six aligned column arrays returned to Python
#[pyfunction]
fn space_weather_days(
    csv_text: &str,
) -> (Vec<i32>, Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let days = parse_space_weather(csv_text);
    (
        days.iter().map(|d| d.year).collect(),
        days.iter().map(|d| d.month).collect(),
        days.iter().map(|d| d.day).collect(),
        days.iter().map(|d| d.ap).collect(),
        days.iter().map(|d| d.f107_observed).collect(),
        days.iter().map(|d| d.f107_center81).collect(),
    )
}

/// The nonlinear density-regression driver features for one day's space weather:
/// `[(F+F81)/2, Ap, ((F+F81)/2)^2, sqrt(Ap), 1]`. Feed rows of these to `ReducedDynamics` for the
/// ~0.975-R^2 density fit (vs ~0.955 for bare linear F10.7).
#[pyfunction]
fn space_weather_drivers(f107: f64, f107_81: f64, ap: f64) -> Vec<f64> {
    density_drivers(f107, f107_81, ap).to_vec()
}

/// The seven-element NRLMSISE-00 storm-time `ap_array` for `year-month-day` at `sec` UT seconds into
/// the day, built from the 3-hourly `AP1..AP8` history in a CelesTrak `SW-All.csv`. Layout:
/// `[daily Ap, current 3 h ap, -3 h, -6 h, -9 h, mean(-12..-33 h), mean(-36..-57 h)]`. Returns `None`
/// if the day or its ~2.5-day look-back is missing. Feed this to the model's array-Ap mode (switch
/// 9 = -1) to resolve the storm-time density surge the daily `Ap` smooths out.
#[pyfunction]
fn space_weather_ap_array(
    csv_text: &str,
    year: i32,
    month: u32,
    day: u32,
    sec: f64,
) -> Option<Vec<f64>> {
    let days = parse_space_weather(csv_text);
    nrlmsise_ap_array(&days, year, month, day, sec).map(|a| a.to_vec())
}

/// The physical free-molecular (Sentman) drag coefficients `(c_p, c_tau, c_d, c_l)` for a flat panel
/// at speed ratio `speed_ratio`, incidence `cos_incidence = cos(delta)` (windward when >= 0), energy
/// accommodation `accommodation` (0..1), wall temperature `wall_temperature_k` and ambient
/// `atmosphere_temperature_k`. This is the physics of a variable drag coefficient, vs a fixed ~2.2.
#[pyfunction]
fn sentman_drag_coefficients(
    speed_ratio: f64,
    cos_incidence: f64,
    accommodation: f64,
    wall_temperature_k: f64,
    atmosphere_temperature_k: f64,
) -> (f64, f64, f64, f64) {
    let c = sentman_panel(
        speed_ratio,
        cos_incidence,
        accommodation,
        wall_temperature_k,
        atmosphere_temperature_k,
    );
    (c.pressure, c.shear, c.drag, c.lift)
}

/// The molecular speed ratio `s = v_rel / sqrt(2 R T / M)` from relative speed `relative_speed_ms`
/// (m/s), ambient temperature `temperature_k` (K) and mean molar mass `mean_molar_mass` (kg/mol).
#[pyfunction]
fn molecular_speed_ratio(relative_speed_ms: f64, temperature_k: f64, mean_molar_mass: f64) -> f64 {
    sentman_speed_ratio(relative_speed_ms, temperature_k, mean_molar_mass)
}

/// IERS 2010 conventional secular corrections `(dC20, dC30, dC40)` to the normalized low-degree zonal
/// geopotential coefficients at `years_since_j2000` decimal years after J2000.0 — add to the base
/// field's C20/C30/C40 for time-variable gravity.
#[pyfunction]
fn time_variable_zonal_correction(years_since_j2000: f64) -> (f64, f64, f64) {
    let c = tvg_secular(years_since_j2000);
    (c[0], c[1], c[2])
}

/// The full IERS 2010 conventional zonal coefficients `(C20, C30, C40)` at `years_since_j2000` (mean +
/// secular drift; C20 zero-tide).
#[pyfunction]
fn conventional_zonal_coefficients(years_since_j2000: f64) -> (f64, f64, f64) {
    let c = tvg_conventional(years_since_j2000);
    (c[0], c[1], c[2])
}

/// Height-normalized absolute log-density from a TU Delft density file: for each nominal observation,
/// `ln ρ` lifted to `reference_altitude_m` via exponential `scale_height_m`. Returns aligned arrays
/// `(years, months, days, latitudes_rad, local_solar_times_h, ln_rho_ref)`. Unlike
/// `density_data_samples` this keeps the absolute amplitude — the signal that tracks solar flux.
#[allow(clippy::type_complexity)] // six aligned column arrays returned to Python
#[pyfunction]
fn density_altitude_normalized(
    text: &str,
    reference_altitude_m: f64,
    scale_height_m: f64,
) -> (Vec<i32>, Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut years = Vec::new();
    let mut months = Vec::new();
    let mut days = Vec::new();
    let mut latitudes = Vec::new();
    let mut local_solar_times = Vec::new();
    let mut ln_rho_ref = Vec::new();
    for o in &parse_density(text) {
        if o.nominal && o.density > 0.0 {
            years.push(o.year);
            months.push(o.month);
            days.push(o.day);
            latitudes.push(o.latitude);
            local_solar_times.push(o.local_solar_time);
            ln_rho_ref.push(o.normalized_log_density(reference_altitude_m, scale_height_m));
        }
    }
    (
        years,
        months,
        days,
        latitudes,
        local_solar_times,
        ln_rho_ref,
    )
}

/// A POD reduced-order density model (ML-ROM): compress field snapshots to the top energetic spatial
/// modes, then project/reconstruct. Fit from openly published density fields (CHAMP/GRACE/Swarm).
#[pyclass(name = "DensityRom")]
struct PyDensityRom {
    inner: DensityRom,
}

#[pymethods]
impl PyDensityRom {
    /// Fit a POD basis to `snapshots` (a list of equal-length field samples, e.g. log-density over a
    /// grid), keeping the top `rank` most energetic modes.
    #[staticmethod]
    fn fit(snapshots: Vec<Vec<f64>>, rank: usize) -> PyResult<Self> {
        DensityRom::fit(&snapshots, rank)
            .map(|inner| Self { inner })
            .ok_or_else(|| {
                PyValueError::new_err(
                    "DensityRom.fit: need >=2 equal-length snapshots and 0 < rank <= snapshots",
                )
            })
    }

    /// The number of retained modes.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// The field dimension `n`.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// The fraction of total field variance captured by the retained modes, in `[0, 1]`.
    #[getter]
    fn energy_fraction(&self) -> f64 {
        self.inner.energy_fraction()
    }

    /// The mean field.
    fn mean(&self) -> Vec<f64> {
        self.inner.mean().to_vec()
    }

    /// Project a full field onto the reduced coordinates.
    fn project(&self, field: Vec<f64>) -> Vec<f64> {
        self.inner.project(&field)
    }

    /// Reconstruct a full field from reduced coordinates.
    fn reconstruct(&self, coords: Vec<f64>) -> Vec<f64> {
        self.inner.reconstruct(&coords)
    }
}

/// A learned linear map from space-weather drivers to POD reduced coordinates — the ML-ROM dynamics
/// layer: `alpha = W . drivers`.
#[pyclass(name = "ReducedDynamics")]
struct PyReducedDynamics {
    inner: ReducedDynamics,
}

#[pymethods]
impl PyReducedDynamics {
    /// Fit `W` so `coord_rows[k] ~= W . driver_rows[k]` by least squares. Add a constant `1.0` driver
    /// per row for an intercept.
    #[staticmethod]
    fn fit(driver_rows: Vec<Vec<f64>>, coord_rows: Vec<Vec<f64>>) -> PyResult<Self> {
        ReducedDynamics::fit(&driver_rows, &coord_rows)
            .map(|inner| Self { inner })
            .ok_or_else(|| {
                PyValueError::new_err(
                    "ReducedDynamics.fit: shape mismatch or singular normal-equations system",
                )
            })
    }

    /// Predict the reduced coordinates for a driver row.
    fn predict(&self, drivers: Vec<f64>) -> Vec<f64> {
        self.inner.predict(&drivers)
    }
}

/// DMDc reduced-order linear dynamics `a_{k+1} = A a_k + B u_k` for the density ROM — the
/// dynamic (with-memory) upgrade over the static `ReducedDynamics` driver->coordinate map.
#[pyclass(name = "Dmdc")]
struct PyDmdc {
    inner: DmdcModel,
}

#[pymethods]
impl PyDmdc {
    /// Fit `[A | B]` so `next_coords[k] ~= A . prev_coords[k] + B . controls[k]` by least squares.
    /// Add a constant `1.0` control per row for an affine offset.
    #[staticmethod]
    fn fit(
        prev_coords: Vec<Vec<f64>>,
        controls: Vec<Vec<f64>>,
        next_coords: Vec<Vec<f64>>,
    ) -> PyResult<Self> {
        DmdcModel::fit(&prev_coords, &controls, &next_coords)
            .map(|inner| Self { inner })
            .ok_or_else(|| {
                PyValueError::new_err(
                    "Dmdc.fit: shape mismatch or singular normal-equations system",
                )
            })
    }

    /// The reduced state one step ahead: `A . state + B . control`.
    fn predict_next(&self, state: Vec<f64>, control: Vec<f64>) -> Vec<f64> {
        self.inner.predict_next(&state, &control)
    }

    /// The free-dynamics operator `A` (r x r); its spectrum is the DMD growth/decay modes.
    fn system_matrix(&self) -> Vec<Vec<f64>> {
        self.inner.system_matrix()
    }

    /// The control operator `B` (r x p): how the drivers force the reduced coordinates.
    fn control_matrix(&self) -> Vec<Vec<f64>> {
        self.inner.control_matrix()
    }
}

/// The full time-variable gravity field parsed from an ICGEM `gfc` model (EIGEN-6S / ITSG-Grace):
/// every degree/order's secular trend plus annual/semiannual terms, evaluated at a decimal year.
#[pyclass(name = "TimeVariableGravity")]
struct PyTimeVariableGravity {
    inner: TimeVariableField,
}

#[pymethods]
impl PyTimeVariableGravity {
    /// Parse the time-variable part (gfct/trnd/acos/asin) of an ICGEM gfc model file's text.
    #[staticmethod]
    fn parse(text: &str) -> Self {
        Self {
            inner: parse_icgem_time_variable(text),
        }
    }

    /// The time-variable `(C, S)` of coefficient `(degree, order)` at decimal `year`, or `None` if it
    /// has no time-variable entry (then it is purely the static base field).
    fn coefficient(&self, degree: usize, order: usize, year: f64) -> Option<(f64, f64)> {
        self.inner.coefficient(degree, order, year)
    }

    /// The number of time-variable coefficients in the model.
    fn __len__(&self) -> usize {
        self.inner.coefficients.len()
    }
}

/// The GLONASS FDMA L1 carrier frequency (Hz) for channel `k`: `(1602.0 + k·0.5625)` MHz.
#[pyfunction]
fn glonass_l1_frequency(channel: i8) -> f64 {
    gnss_glonass_l1(channel)
}

/// The GLONASS FDMA L2 carrier frequency (Hz) for channel `k`: `(1246.0 + k·0.4375)` MHz.
#[pyfunction]
fn glonass_l2_frequency(channel: i8) -> f64 {
    gnss_glonass_l2(channel)
}

/// Sensor swath geometry for an off-nadir `look_angle` (rad) at `altitude` (m) over a spherical Earth
/// of radius `re` (m): `(earth_central_angle, elevation, ground_range, slant_range)`, or `None` past the
/// limb.
#[pyfunction]
fn swath_geometry(re: f64, altitude: f64, look_angle: f64) -> Option<(f64, f64, f64, f64)> {
    ev_swath_geometry(re, altitude, look_angle).map(|g: SwathGeometry| {
        (
            g.earth_central_angle,
            g.elevation,
            g.ground_range,
            g.slant_range,
        )
    })
}

/// The limb (maximum) off-nadir look angle whose line of sight still meets the Earth, radians.
#[pyfunction]
fn horizon_look_angle(re: f64, altitude: f64) -> f64 {
    ev_horizon_look_angle(re, altitude)
}

/// The geodetic `(lat, lon)` (rad) reached by a great-circle arc of `central_angle` (rad) from
/// `(lat, lon)` on the initial `azimuth` (rad). Sweep the azimuth to trace a footprint circle.
#[pyfunction]
fn ground_point_at(lat: f64, lon: f64, central_angle: f64, azimuth: f64) -> (f64, f64) {
    ev_ground_point_at(lat, lon, central_angle, azimuth)
}

/// An RPC00B rational-polynomial camera. Build with `Rpc.parse_rpb(text)` from a `.RPB` file's text;
/// `forward`/`inverse` are the ground↔image maps (lat/lon in degrees, height/metres, image in pixels).
#[pyclass(name = "Rpc")]
struct PyRpc {
    inner: RationalPolynomialCamera,
}

#[pymethods]
impl PyRpc {
    /// Parse an RPC from `.RPB` file text; raises `ValueError` if a required field is missing.
    #[staticmethod]
    fn parse_rpb(text: &str) -> PyResult<Self> {
        RationalPolynomialCamera::parse_rpb(text)
            .map(|inner| Self { inner })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("invalid or incomplete .RPB text")
            })
    }

    /// Project ground `(latitude, longitude)` degrees at `height` m to image `(row, col)` pixels.
    fn forward(&self, latitude: f64, longitude: f64, height: f64) -> (f64, f64) {
        self.inner.forward(latitude, longitude, height)
    }

    /// Solve for the ground `(latitude, longitude)` degrees imaging to `(row, col)` at `height` m.
    ///
    /// Raises `ValueError` if the Newton solve finds no ground point — a pixel outside the image the
    /// RPC was fitted for has none.
    fn inverse(&self, row: f64, col: f64, height: f64) -> PyResult<(f64, f64)> {
        let g = self
            .inner
            .inverse(row, col, height)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok((g.latitude, g.longitude))
    }
}

/// 6S atmospheric correction — surface reflectance behind a measured `toa_reflectance`, given the
/// atmosphere's path reflectance, downward/upward transmittances and spherical albedo.
#[pyfunction]
fn atmospheric_surface_from_toa(
    path_reflectance: f64,
    transmittance_down: f64,
    transmittance_up: f64,
    spherical_albedo: f64,
    toa_reflectance: f64,
) -> f64 {
    AtmosphericCoupling {
        path_reflectance,
        transmittance_down,
        transmittance_up,
        spherical_albedo,
    }
    .surface_from_toa(toa_reflectance)
}

/// 6S forward — the TOA reflectance a surface of `surface_reflectance` produces under the given
/// atmosphere terms (inverse of [`atmospheric_surface_from_toa`]).
#[pyfunction]
fn atmospheric_toa_from_surface(
    path_reflectance: f64,
    transmittance_down: f64,
    transmittance_up: f64,
    spherical_albedo: f64,
    surface_reflectance: f64,
) -> f64 {
    AtmosphericCoupling {
        path_reflectance,
        transmittance_down,
        transmittance_up,
        spherical_albedo,
    }
    .toa_from_surface(surface_reflectance)
}

/// MODIS RTLSR surface reflectance `f_iso + f_vol·K_vol + f_geo·K_geo` at `(theta_s, theta_v,
/// rel_azimuth)` radians.
#[pyfunction]
fn brdf_reflectance(
    f_iso: f64,
    f_vol: f64,
    f_geo: f64,
    theta_s: f64,
    theta_v: f64,
    rel_azimuth: f64,
) -> f64 {
    ev_brdf(f_iso, f_vol, f_geo, theta_s, theta_v, rel_azimuth)
}

/// The RossThick volumetric BRDF kernel at `(theta_s, theta_v, rel_azimuth)` radians.
#[pyfunction]
fn ross_thick_kernel(theta_s: f64, theta_v: f64, rel_azimuth: f64) -> f64 {
    ev_ross_thick(theta_s, theta_v, rel_azimuth)
}

/// The LiSparse-Reciprocal geometric BRDF kernel (MODIS `h/b=2`, `b/r=1`) at radians.
#[pyfunction]
fn li_sparse_reciprocal_kernel(theta_s: f64, theta_v: f64, rel_azimuth: f64) -> f64 {
    ev_li_sparse(theta_s, theta_v, rel_azimuth)
}

// ---------------------------------------------------------------------------------------------------
// Phase-8 spacesignal — SAR focusing, InSAR, TomoSAR (complex arrays cross as parallel re/im lists).
// ---------------------------------------------------------------------------------------------------

fn to_sig(re: &[f64], im: &[f64]) -> Vec<SigComplex> {
    re.iter()
        .zip(im)
        .map(|(&r, &i)| SigComplex::new(r, i))
        .collect()
}

fn from_sig(z: &[SigComplex]) -> (Vec<f64>, Vec<f64>) {
    (
        z.iter().map(|c| c.re).collect(),
        z.iter().map(|c| c.im).collect(),
    )
}

/// Wrap a phase (radians) to `(−π, π]`.
#[pyfunction]
fn wrap_phase(phi: f64) -> f64 {
    sig_wrap_phase(phi)
}

/// The Goldstein residue map of a wrapped field (`rows × cols`, row-major), as a `(rows−1)·(cols−1)`
/// list of `+1/−1/0` charges.
#[pyfunction]
fn residues(wrapped: Vec<f64>, rows: usize, cols: usize) -> Vec<i32> {
    sig_residues(&wrapped, rows, cols)
        .into_iter()
        .map(i32::from)
        .collect()
}

/// 1-D Itoh phase unwrapping.
#[pyfunction]
fn unwrap_phase_1d(wrapped: Vec<f64>) -> Vec<f64> {
    sig_unwrap_1d(&wrapped)
}

/// 2-D path-following phase unwrapping (row-major `rows × cols`).
#[pyfunction]
fn unwrap_phase_2d(wrapped: Vec<f64>, rows: usize, cols: usize) -> Vec<f64> {
    sig_unwrap_2d(&wrapped, rows, cols)
}

/// 2-D least-squares (Ghiglia–Romero DCT) phase unwrapping — residue-robust.
#[pyfunction]
fn unwrap_phase_2d_lstsq(wrapped: Vec<f64>, rows: usize, cols: usize) -> Vec<f64> {
    sig_unwrap_lstsq(&wrapped, rows, cols)
}

/// 2-D Goldstein branch-cut phase unwrapping — residue-robust.
#[pyfunction]
fn unwrap_phase_2d_branch_cut(wrapped: Vec<f64>, rows: usize, cols: usize) -> Vec<f64> {
    sig_unwrap_branch_cut(&wrapped, rows, cols)
}

/// The interferogram `s₁·conj(s₂)` of two complex images (re/im lists); returns `(re, im)`.
#[pyfunction]
fn interferogram(
    s1_re: Vec<f64>,
    s1_im: Vec<f64>,
    s2_re: Vec<f64>,
    s2_im: Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    from_sig(&sig_interferogram(
        &to_sig(&s1_re, &s1_im),
        &to_sig(&s2_re, &s2_im),
    ))
}

/// Per-pixel InSAR coherence over a `(2·half_window+1)²` boxcar, in `[0, 1]`.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // two complex images (four lists), shape, and the window
fn coherence(
    s1_re: Vec<f64>,
    s1_im: Vec<f64>,
    s2_re: Vec<f64>,
    s2_im: Vec<f64>,
    rows: usize,
    cols: usize,
    half_window: usize,
) -> Vec<f64> {
    sig_coherence(
        &to_sig(&s1_re, &s1_im),
        &to_sig(&s2_re, &s2_im),
        rows,
        cols,
        half_window,
    )
}

/// A linear-FM chirp of `n` samples (`exp(jπ·chirp_rate·t²)`, `t` centred at `sample_rate`); `(re, im)`.
#[pyfunction]
fn lfm_chirp(n: usize, chirp_rate: f64, sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    from_sig(&sig_lfm_chirp(n, chirp_rate, sample_rate))
}

/// Matched-filter (pulse-compress) a complex `signal` with a complex `replica`; returns `(re, im)`.
#[pyfunction]
fn matched_filter(
    signal_re: Vec<f64>,
    signal_im: Vec<f64>,
    replica_re: Vec<f64>,
    replica_im: Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    from_sig(&sig_matched_filter(
        &to_sig(&signal_re, &signal_im),
        &to_sig(&replica_re, &replica_im),
    ))
}

/// The −3 dB main-lobe width (samples) of a magnitude impulse response. `nan` when there is no
/// width to report: an empty input, a peak not strictly above its own half power, or a main lobe
/// that runs off the end of the array.
#[pyfunction]
fn resolution_3db_samples(mag: Vec<f64>) -> f64 {
    sig_res3db(&mag)
}

/// The peak-sidelobe ratio (dB) of a magnitude impulse response. `nan` for an empty input, which
/// has no peak to be relative to.
#[pyfunction]
fn peak_sidelobe_ratio_db(mag: Vec<f64>) -> f64 {
    sig_pslr(&mag)
}

fn rda_params(
    prf: f64,
    velocity: f64,
    wavelength: f64,
    r0_slant: f64,
    range_spacing: f64,
) -> RdaParams {
    RdaParams {
        prf,
        velocity,
        wavelength,
        r0_slant,
        range_spacing,
    }
}

/// The azimuth FM rate `K_a = 2V²/(λ·r₀)` (Hz/s).
#[pyfunction]
fn azimuth_fm_rate(
    prf: f64,
    velocity: f64,
    wavelength: f64,
    r0_slant: f64,
    range_spacing: f64,
) -> f64 {
    sig_azimuth_fm_rate(&rda_params(
        prf,
        velocity,
        wavelength,
        r0_slant,
        range_spacing,
    ))
}

/// The range-cell migration `ΔR(f_η)` (m) at azimuth frequency `f_eta` (Hz).
#[pyfunction]
#[allow(clippy::too_many_arguments)] // azimuth frequency + the five acquisition parameters
fn range_migration(
    f_eta: f64,
    prf: f64,
    velocity: f64,
    wavelength: f64,
    r0_slant: f64,
    range_spacing: f64,
) -> f64 {
    sig_range_migration(
        f_eta,
        &rda_params(prf, velocity, wavelength, r0_slant, range_spacing),
    )
}

/// The secondary-range-compression FM rate `K_src(f_η)` (Hz/s), or `None` at zero Doppler.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // azimuth frequency + the five acquisition parameters
fn src_fm_rate(
    f_eta: f64,
    prf: f64,
    velocity: f64,
    wavelength: f64,
    r0_slant: f64,
    range_spacing: f64,
) -> Option<f64> {
    sig_src_fm_rate(
        f_eta,
        &rda_params(prf, velocity, wavelength, r0_slant, range_spacing),
    )
}

/// The Bartlett (beamforming) TomoSAR elevation spectrum over `elevations` (m), from an `N×N` complex
/// `covariance` (re/im row-major) for the given `baselines`, `wavelength` and `range`.
#[pyfunction]
fn bartlett_spectrum(
    covariance_re: Vec<f64>,
    covariance_im: Vec<f64>,
    baselines: Vec<f64>,
    wavelength: f64,
    range: f64,
    elevations: Vec<f64>,
) -> Vec<f64> {
    sig_bartlett(
        &to_sig(&covariance_re, &covariance_im),
        &baselines,
        wavelength,
        range,
        &elevations,
    )
}

/// The Capon (MVDR) TomoSAR elevation spectrum, with diagonal `loading`.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // complex covariance (two lists), geometry, grid, and loading
fn capon_spectrum(
    covariance_re: Vec<f64>,
    covariance_im: Vec<f64>,
    baselines: Vec<f64>,
    wavelength: f64,
    range: f64,
    elevations: Vec<f64>,
    loading: f64,
) -> Vec<f64> {
    sig_capon(
        &to_sig(&covariance_re, &covariance_im),
        &baselines,
        wavelength,
        range,
        &elevations,
        loading,
    )
}

/// SL1MMER ℓ₁ elevation spectrum `|γ|` from a single complex stack `y` (re/im).
#[pyfunction]
#[allow(clippy::too_many_arguments)] // complex measurement (two lists), geometry, grid, and solver knobs
fn sl1mmer_spectrum(
    y_re: Vec<f64>,
    y_im: Vec<f64>,
    baselines: Vec<f64>,
    wavelength: f64,
    range: f64,
    elevations: Vec<f64>,
    lambda: f64,
    max_iter: usize,
) -> Vec<f64> {
    sig_sl1mmer_spectrum(
        &to_sig(&y_re, &y_im),
        &baselines,
        wavelength,
        range,
        &elevations,
        lambda,
        max_iter,
    )
}

/// Full SL1MMER: returns the detected scatterers as `(elevation, reflectivity_re, reflectivity_im)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // complex measurement (two lists), geometry, grid, and solver knobs
fn sl1mmer(
    y_re: Vec<f64>,
    y_im: Vec<f64>,
    baselines: Vec<f64>,
    wavelength: f64,
    range: f64,
    elevations: Vec<f64>,
    lambda: f64,
    max_iter: usize,
    threshold_rel: f64,
) -> Vec<(f64, f64, f64)> {
    let cfg = Sl1mmerConfig {
        lambda,
        max_iter,
        threshold_rel,
    };
    sig_sl1mmer(
        &to_sig(&y_re, &y_im),
        &baselines,
        wavelength,
        range,
        &elevations,
        &cfg,
    )
    .into_iter()
    .map(|s| (s.elevation, s.reflectivity.re, s.reflectivity.im))
    .collect()
}

// ---------------------------------------------------------------------------------------------------
// Phase-8 earthvision v2–v4 — SGM stereo, Harris corners, RPC block adjustment.
// ---------------------------------------------------------------------------------------------------

/// Dense stereo disparity by Semi-Global Matching of a rectified `left`/`right` pair (`h × w`,
/// row-major grayscale). Returns one disparity per pixel.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // the two images, the shape, and the four SGM knobs
fn semi_global_match(
    left: Vec<f64>,
    right: Vec<f64>,
    h: usize,
    w: usize,
    d_max: usize,
    radius: usize,
    p1: f64,
    p2: f64,
) -> Vec<f64> {
    let cfg = SgmConfig {
        d_max,
        radius,
        p1,
        p2,
    };
    ev_sgm(&left, &right, h, w, &cfg)
}

/// Detect Harris corners of an `h × w` grayscale image; returns `(x, y, response)` per corner.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // the image, its shape, and the five detector knobs
fn harris_corners(
    image: Vec<f64>,
    h: usize,
    w: usize,
    k: f64,
    window_radius: usize,
    threshold_rel: f64,
    nms_radius: usize,
    max_corners: usize,
) -> Vec<(f64, f64, f64)> {
    let img = Image { data: &image, h, w };
    let cfg = HarrisConfig {
        k,
        window_radius,
        threshold_rel,
        nms_radius,
        max_corners,
    };
    ev_detect_corners(img, &cfg)
        .into_iter()
        .map(|c| (c.x, c.y, c.response))
        .collect()
}

/// Estimate a Grodecki–Dial RPC bias correction from ground control points `(lat, lon, height, row,
/// col)` and apply it: returns the corrected image `(row, col)` for `query = (lat, lon, height)`.
/// `affine = True` fits the six-parameter affine, else the two shifts.
#[pyfunction]
fn rpc_block_adjust(
    rpc: &PyRpc,
    gcps: Vec<(f64, f64, f64, f64, f64)>,
    query: (f64, f64, f64),
    affine: bool,
) -> (f64, f64) {
    let points: Vec<GroundControlPoint> = gcps
        .into_iter()
        .map(
            |(latitude, longitude, height, row, col)| GroundControlPoint {
                latitude,
                longitude,
                height,
                row,
                col,
            },
        )
        .collect();
    let model = if affine {
        AdjustmentModel::Affine
    } else {
        AdjustmentModel::Shift
    };
    let adj: RpcAdjustment = ev_estimate_adjustment(&rpc.inner, &points, model);
    ev_forward_adjusted(&rpc.inner, query.0, query.1, query.2, &adj)
}

// ---------------------------------------------------------------------------------------------------
// Phase-8 deepvision — object-detection head (anchors, box decode, IoU, NMS).
// ---------------------------------------------------------------------------------------------------

/// Greedy non-maximum suppression of axis-aligned boxes `(x1, y1, x2, y2)` with `scores`, at
/// `iou_threshold`. Returns the kept indices (descending score).
#[pyfunction]
fn nms(boxes: Vec<(f64, f64, f64, f64)>, scores: Vec<f64>, iou_threshold: f64) -> Vec<usize> {
    let bs: Vec<BBox> = boxes
        .into_iter()
        .map(|(x1, y1, x2, y2)| BBox::new(x1, y1, x2, y2))
        .collect();
    dv_nms(&bs, &scores, iou_threshold)
}

/// Intersection-over-union of two boxes `(x1, y1, x2, y2)`.
#[pyfunction]
fn box_iou(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> f64 {
    BBox::new(a.0, a.1, a.2, a.3).iou(&BBox::new(b.0, b.1, b.2, b.3))
}

/// Lay anchor boxes over a `feat_h × feat_w` feature map at pixel `stride`, for the `scales` and
/// `aspect_ratios`. Returns `(x1, y1, x2, y2)` per anchor.
#[pyfunction]
fn generate_anchors(
    feat_h: usize,
    feat_w: usize,
    stride: f64,
    scales: Vec<f64>,
    aspect_ratios: Vec<f64>,
) -> Vec<(f64, f64, f64, f64)> {
    dv_generate_anchors(feat_h, feat_w, stride, &scales, &aspect_ratios)
        .into_iter()
        .map(|b| (b.x1, b.y1, b.x2, b.y2))
        .collect()
}

/// Decode a predicted box delta `(dx, dy, dw, dh)` against an `anchor` (Faster-R-CNN centre form).
#[pyfunction]
fn decode_box(anchor: (f64, f64, f64, f64), delta: (f64, f64, f64, f64)) -> (f64, f64, f64, f64) {
    let b = dv_decode_box(
        &BBox::new(anchor.0, anchor.1, anchor.2, anchor.3),
        [delta.0, delta.1, delta.2, delta.3],
    );
    (b.x1, b.y1, b.x2, b.y2)
}

// ---------------------------------------------------------------------------------------------------
// Phase-8 v5 — multi-view geometry (camera, triangulate, MVS, SfM), pan-sharpening, PolSAR, PS/SBAS.
// ---------------------------------------------------------------------------------------------------

/// A calibrated pinhole camera: rotation (world→camera, 9 row-major), centre (world, 3), focal length
/// and principal point (pixels). Projects world points to pixels via the collinearity equations.
#[pyclass(name = "PinholeCamera", from_py_object)]
#[derive(Clone, Copy)]
struct PyCamera {
    inner: EvPinholeCamera,
}

#[pymethods]
impl PyCamera {
    #[new]
    fn new(rotation: [f64; 9], center: [f64; 3], focal: f64, principal: [f64; 2]) -> Self {
        Self {
            inner: EvPinholeCamera {
                rotation: Mat3::from_rows([
                    [rotation[0], rotation[1], rotation[2]],
                    [rotation[3], rotation[4], rotation[5]],
                    [rotation[6], rotation[7], rotation[8]],
                ]),
                center: Vec3::new(center[0], center[1], center[2]),
                focal,
                principal,
            },
        }
    }

    /// Project a world point `(x, y, z)` to a pixel `(u, v)`.
    fn project(&self, world: (f64, f64, f64)) -> (f64, f64) {
        let p = self.inner.project(Vec3::new(world.0, world.1, world.2));
        (p[0], p[1])
    }
}

/// Triangulate a world point from `cameras` and their `pixels` (≥2 views); `None` if degenerate.
#[pyfunction]
fn triangulate(cameras: Vec<PyCamera>, pixels: Vec<(f64, f64)>) -> Option<(f64, f64, f64)> {
    let cams: Vec<EvPinholeCamera> = cameras.iter().map(|c| c.inner).collect();
    let views: Vec<(&EvPinholeCamera, [f64; 2])> = cams
        .iter()
        .zip(&pixels)
        .map(|(c, &(u, v))| (c, [u, v]))
        .collect();
    ev_triangulate(&views).map(|p| (p.x, p.y, p.z))
}

/// Dense depth map by plane-sweep MVS of a `reference` image (taken by `ref_cam`) against one
/// `neighbor` (taken by `nbr_cam`), sweeping `depths` with a `(2·patch_radius+1)²` NCC window.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // two images + two cameras, the shape, and the sweep parameters
fn plane_sweep_mvs(
    reference: Vec<f64>,
    ref_cam: &PyCamera,
    neighbor: Vec<f64>,
    nbr_cam: &PyCamera,
    h: usize,
    w: usize,
    depths: Vec<f64>,
    patch_radius: usize,
) -> Vec<f64> {
    ev_plane_sweep(
        &reference,
        &ref_cam.inner,
        &[(&neighbor, &nbr_cam.inner)],
        h,
        w,
        &depths,
        patch_radius,
    )
}

/// Build SfM feature tracks from per-view corners `(x, y, response)` and `(view_a, corner_a, view_b,
/// corner_b)` matches, then triangulate each with `cameras`. Returns `(x, y, z)` per track (or `None`).
#[pyfunction]
fn sfm_triangulate(
    corners_per_view: Vec<Vec<(f64, f64, f64)>>,
    matches: Vec<(usize, usize, usize, usize)>,
    cameras: Vec<PyCamera>,
) -> Vec<Option<(f64, f64, f64)>> {
    let corner_lists: Vec<Vec<Corner>> = corners_per_view
        .iter()
        .map(|cs| {
            cs.iter()
                .map(|&(x, y, response)| Corner { x, y, response })
                .collect()
        })
        .collect();
    let refs: Vec<&[Corner]> = corner_lists.iter().map(Vec::as_slice).collect();
    let tracks: Vec<Track> = ev_build_tracks(&refs, &matches);
    let cams: Vec<EvPinholeCamera> = cameras.iter().map(|c| c.inner).collect();
    tracks
        .iter()
        .map(|t| {
            let views: Vec<(&EvPinholeCamera, [f64; 2])> = t
                .observations
                .iter()
                .map(|o: &Observation2d| (&cams[o.view], [o.x, o.y]))
                .collect();
            ev_triangulate(&views).map(|p| (p.x, p.y, p.z))
        })
        .collect()
}

/// Brovey pan-sharpen `ms_bands` (each a `pixels`-length band) with `pan`. Returns one band per input.
#[pyfunction]
fn brovey_pan_sharpen(ms_bands: Vec<Vec<f64>>, pan: Vec<f64>) -> Vec<Vec<f64>> {
    let refs: Vec<&[f64]> = ms_bands.iter().map(Vec::as_slice).collect();
    ev_brovey(&refs, &pan)
}

/// Gram–Schmidt component-substitution pan-sharpen `ms_bands` with `pan`.
#[pyfunction]
fn gram_schmidt_pan_sharpen(ms_bands: Vec<Vec<f64>>, pan: Vec<f64>) -> Vec<Vec<f64>> {
    let refs: Vec<&[f64]> = ms_bands.iter().map(Vec::as_slice).collect();
    ev_gram_schmidt(&refs, &pan)
}

/// The Pauli bounce powers `[odd, even, volume]` of a scattering matrix (each element a complex
/// `(re, im)`).
#[pyfunction]
fn pauli_powers(s_hh: (f64, f64), s_hv: (f64, f64), s_vv: (f64, f64)) -> (f64, f64, f64) {
    let p = sig_pauli_powers(
        SigComplex::new(s_hh.0, s_hh.1),
        SigComplex::new(s_hv.0, s_hv.1),
        SigComplex::new(s_vv.0, s_vv.1),
    );
    (p[0], p[1], p[2])
}

/// The `3×3` Hermitian coherency matrix `T₃` from scattering `samples` (each `(hh_re, hh_im, hv_re,
/// hv_im, vv_re, vv_im)`); returns `(re, im)` of the 9 row-major elements.
#[pyfunction]
fn coherency_matrix(samples: Vec<(f64, f64, f64, f64, f64, f64)>) -> (Vec<f64>, Vec<f64>) {
    let s: Vec<(SigComplex, SigComplex, SigComplex)> = samples
        .into_iter()
        .map(|(a, b, c, d, e, f)| {
            (
                SigComplex::new(a, b),
                SigComplex::new(c, d),
                SigComplex::new(e, f),
            )
        })
        .collect();
    from_sig(&sig_coherency(&s))
}

/// The Cloude–Pottier `(entropy, anisotropy, mean_alpha)` of a `3×3` Hermitian coherency matrix (re/im
/// row-major); `mean_alpha` in radians.
#[pyfunction]
fn cloude_pottier(t3_re: Vec<f64>, t3_im: Vec<f64>) -> (f64, f64, f64) {
    sig_cloude_pottier(&to_sig(&t3_re, &t3_im))
}

/// Amplitude dispersion `D_A = σ_A/μ_A` of every pixel over a `stack` of `pixels`-length amplitude
/// images — the PS-InSAR phase-stability proxy.
#[pyfunction]
fn amplitude_dispersion(stack: Vec<Vec<f64>>, pixels: usize) -> Vec<f64> {
    let refs: Vec<&[f64]> = stack.iter().map(Vec::as_slice).collect();
    sig_amplitude_dispersion(&refs, pixels)
}

/// Indices whose amplitude `dispersion` is below `threshold` — the persistent scatterers.
#[pyfunction]
fn select_persistent_scatterers(dispersion: Vec<f64>, threshold: f64) -> Vec<usize> {
    sig_select_ps(&dispersion, threshold)
}

/// SBAS inversion of one scatterer's unwrapped `phases` for `(velocity, dem_error)` over the temporal
/// and perpendicular baselines. `wavelength`/`range` metres, `incidence` radians.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // phases + two baseline lists + the three geometry scalars
fn sbas_invert(
    phases: Vec<f64>,
    temporal_baselines: Vec<f64>,
    perp_baselines: Vec<f64>,
    wavelength: f64,
    range: f64,
    incidence: f64,
) -> (f64, f64) {
    sig_sbas_invert(
        &phases,
        &temporal_baselines,
        &perp_baselines,
        wavelength,
        range,
        incidence,
    )
}

/// Focus a raw SAR echo (`na × nr`, row-major complex re/im) by the interpolation-free Chirp-Scaling
/// Algorithm. Returns the focused image `(re, im)`. `na`, `nr` powers of two.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // the complex echo, its shape, and the six acquisition parameters
fn chirp_scaling_focus(
    raw_re: Vec<f64>,
    raw_im: Vec<f64>,
    na: usize,
    nr: usize,
    prf: f64,
    velocity: f64,
    wavelength: f64,
    r_ref: f64,
    range_sample_rate: f64,
    range_fm_rate: f64,
) -> (Vec<f64>, Vec<f64>) {
    let p = CsaParams {
        prf,
        velocity,
        wavelength,
        r_ref,
        range_sample_rate,
        range_fm_rate,
    };
    from_sig(&sig_csa_focus(&to_sig(&raw_re, &raw_im), na, nr, &p))
}

/// The chirp-scaling phase multiply on a complex range `line` (re/im): `exp(jπ·chirp_rate·scaling·(t −
/// t_ref)²)`, `t` centred on sample `tau_ref` at `sample_rate`. Returns `(re, im)`.
#[pyfunction]
fn apply_chirp_scaling(
    line_re: Vec<f64>,
    line_im: Vec<f64>,
    chirp_rate: f64,
    scaling: f64,
    tau_ref: f64,
    sample_rate: f64,
) -> (Vec<f64>, Vec<f64>) {
    from_sig(&sig_apply_chirp_scaling(
        &to_sig(&line_re, &line_im),
        chirp_rate,
        scaling,
        tau_ref,
        sample_rate,
    ))
}

/// Run a cloud-segmentation ONNX model over a `channels x height x width` tile and return the flat
/// per-pixel cloud mask (`True` = cloudy) over the model's output grid. `onnx_bytes` is a serialized
/// `.onnx` model. Only present when the extension is built with `--features onnx`.
#[cfg(feature = "onnx")]
#[pyfunction]
fn cloud_mask(
    onnx_bytes: Vec<u8>,
    tile: Vec<f32>,
    channels: usize,
    height: usize,
    width: usize,
    threshold: f32,
) -> PyResult<Vec<bool>> {
    let masker = CloudMasker::new(&onnx_bytes, threshold)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    masker
        .mask(&tile, channels, height, width)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}

// ---------------------------------------------------------------------------------------------
// Multistatic observables (#314) and the deep-space radiometric ones (#324).
//
// Both crates' entry points are plain functions over `Scalar`, so the binding is a transcription
// rather than a design: the states cross as length-6 lists, the geometries are built from their
// terminals here, and every construction that can be refused — a non-causal leg, an out-of-order
// ramp table, a line of sight inside the Sun — is refused as a `ValueError` naming the input.
// ---------------------------------------------------------------------------------------------

/// A workspace error, as the `ValueError` a Python caller sees.
///
/// Every error type this module maps names the argument it refused and its value in its `Display`,
/// so the message is passed through rather than replaced with one of this module's own.
fn value_err<E: core::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(format!("{error}"))
}

/// Bistatic range `ρ_tx + ρ_rx`, metres — one transmitter, one reflecting satellite, one receiver.
///
/// `transmitter` and `receiver` are `(position, velocity, epoch)`; `bounce` is the epoch of the
/// reflection, which is the epoch `satellite` (a length-6 state) belongs to. Raises `ValueError` if
/// the three epochs are not strictly increasing along the signal path.
#[pyfunction]
fn bistatic_range(
    satellite: [f64; 6],
    transmitter: ([f64; 3], [f64; 3], f64),
    bounce: f64,
    receiver: ([f64; 3], [f64; 3], f64),
) -> PyResult<f64> {
    let geometry = bistatic_geometry(transmitter, bounce, receiver)?;
    Ok(estimation_bistatic_range(&satellite, &geometry))
}

/// Bistatic range rate, m/s — the time derivative of [`bistatic_range`], which is what a bistatic
/// Doppler measures once scaled by the carrier. Both terminals contribute their own closing rate.
#[pyfunction]
fn bistatic_range_rate(
    satellite: [f64; 6],
    transmitter: ([f64; 3], [f64; 3], f64),
    bounce: f64,
    receiver: ([f64; 3], [f64; 3], f64),
) -> PyResult<f64> {
    let geometry = bistatic_geometry(transmitter, bounce, receiver)?;
    Ok(estimation_bistatic_range_rate(&satellite, &geometry))
}

/// Time difference of arrival `(ρ_second − ρ_first)/c`, seconds.
///
/// `emission` is the epoch the satellite state belongs to; each terminal is
/// `(position, velocity, epoch)` and its epoch must be strictly after the emission.
#[pyfunction]
fn tdoa(
    satellite: [f64; 6],
    emission: f64,
    first: ([f64; 3], [f64; 3], f64),
    second: ([f64; 3], [f64; 3], f64),
) -> PyResult<f64> {
    let geometry = tdoa_geometry(emission, first, second)?;
    Ok(estimation_tdoa(&satellite, &geometry))
}

/// Frequency difference of arrival `(f/c)(ρ̇_first − ρ̇_second)`, hertz, at carrier `frequency`.
#[pyfunction]
fn fdoa(
    satellite: [f64; 6],
    emission: f64,
    first: ([f64; 3], [f64; 3], f64),
    second: ([f64; 3], [f64; 3], f64),
    frequency: f64,
) -> PyResult<f64> {
    let geometry = tdoa_geometry(emission, first, second)?;
    Ok(estimation_fdoa(&satellite, &geometry, frequency))
}

/// Separation of two satellites, metres — the inter-satellite link's one-way range.
#[pyfunction]
fn inter_satellite_range(local: [f64; 6], remote: [f64; 6]) -> f64 {
    estimation_inter_satellite_range(&local, &remote)
}

/// Inter-satellite range rate, m/s — positive when the two are separating.
#[pyfunction]
fn inter_satellite_range_rate(local: [f64; 6], remote: [f64; 6]) -> f64 {
    estimation_inter_satellite_range_rate(&local, &remote)
}

/// `(position, velocity, epoch)` as a [`Terminal`].
fn terminal(spec: ([f64; 3], [f64; 3], f64)) -> Terminal {
    let (position, velocity, epoch) = spec;
    Terminal::new(Station { position, velocity }, epoch)
}

fn bistatic_geometry(
    transmitter: ([f64; 3], [f64; 3], f64),
    bounce: f64,
    receiver: ([f64; 3], [f64; 3], f64),
) -> PyResult<BistaticGeometry> {
    BistaticGeometry::new(terminal(transmitter), bounce, terminal(receiver)).map_err(value_err)
}

fn tdoa_geometry(
    emission: f64,
    first: ([f64; 3], [f64; 3], f64),
    second: ([f64; 3], [f64; 3], f64),
) -> PyResult<TdoaGeometry> {
    TdoaGeometry::new(emission, terminal(first), terminal(second)).map_err(value_err)
}

/// Moyer's three DSN bands by name, case-insensitively.
///
/// A string rather than an integer, matching the `precision='f64'|'dd'` idiom the rest of this
/// module uses: the caller writes the band the way a tracking schedule does.
fn dsn_band(name: &str) -> PyResult<Band> {
    for (label, band) in [("S", Band::S), ("X", Band::X), ("Ka", Band::Ka)] {
        if name.eq_ignore_ascii_case(label) {
            return Ok(band);
        }
    }
    Err(PyValueError::new_err(format!(
        "band must be 'S', 'X' or 'Ka', not {name:?}"
    )))
}

/// `[(start, end, start_frequency, rate), …]` — seconds, seconds, hertz, hertz per second.
fn ramp_segments(segments: &[(f64, f64, f64, f64)]) -> Vec<RampSegment> {
    segments
        .iter()
        .map(|&(start, end, start_frequency, rate)| RampSegment {
            start,
            end,
            start_frequency,
            rate,
        })
        .collect()
}

/// DSN sequential range in **range units**, wrapped at `2^(C+6)`.
///
/// `segments` is the station's ramp table as `[(start, end, start_frequency, rate), …]`; `transmit`
/// and `receive` are the epochs the ranging code left the transmitting station and arrived at the
/// receiving one, in that table's timescale. `uplink` sets the scale — a range unit is defined
/// against the uplink carrier — and `lowest_ranging_component` is Moyer's `C`.
///
/// Raises `ValueError` on a ramp table that is empty, out of order or does not cover the interval,
/// on an interval that does not run forwards, and on a `C` whose modulus is not representable.
#[pyfunction]
fn dsn_sequential_range(
    segments: Vec<(f64, f64, f64, f64)>,
    uplink: &str,
    transmit: f64,
    receive: f64,
    lowest_ranging_component: i32,
) -> PyResult<f64> {
    let owned = ramp_segments(&segments);
    let table = RampTable::new(&owned).map_err(value_err)?;
    estimation_sequential_range(
        &table,
        dsn_band(uplink)?,
        transmit,
        receive,
        lowest_ranging_component,
    )
    .map_err(value_err)
}

/// DSN n-way averaged Doppler, hertz.
///
/// `M(reference_band, downlink)·reference_frequency − (M(uplink, downlink)/count_interval)·∫f_T`,
/// taken between the two transmission epochs. `reference_band` is carried separately from `uplink`
/// because the receiver's reference need not be stated against the band actually transmitted on;
/// neither has a default, since a wrong one is a silent change of answer rather than an error.
/// `subtracted` selects the DSN convention (the ramp term enters negative), which is Tudat's too.
#[pyfunction]
#[pyo3(signature = (
    segments, uplink, downlink, reference_frequency, reference_band, count_interval,
    transmit_start, transmit_end, subtracted = true,
))]
#[allow(clippy::too_many_arguments)] // the ramp table, three bands, the reference, and the interval
fn dsn_averaged_doppler(
    segments: Vec<(f64, f64, f64, f64)>,
    uplink: &str,
    downlink: &str,
    reference_frequency: f64,
    reference_band: &str,
    count_interval: f64,
    transmit_start: f64,
    transmit_end: f64,
    subtracted: bool,
) -> PyResult<f64> {
    let owned = ramp_segments(&segments);
    let table = RampTable::new(&owned).map_err(value_err)?;
    let link = DopplerLink {
        uplink: dsn_band(uplink)?,
        downlink: dsn_band(downlink)?,
        reference: DopplerReference {
            frequency: reference_frequency,
            band: dsn_band(reference_band)?,
        },
        signature: if subtracted {
            DopplerSignature::Subtracted
        } else {
            DopplerSignature::Added
        },
        count_interval,
    };
    estimation_averaged_doppler(&table, link, transmit_start, transmit_end).map_err(value_err)
}

/// DSN measured frequency, hertz — `M(uplink, downlink)·f_T(transmit_epoch)·transmit_rate`.
///
/// The instantaneous counterpart of [`dsn_averaged_doppler`], and not independent of it: over a
/// flat exciter referred to the same band and frequency, the two sum to `M·f_reference`.
/// `transmit_rate` is `dt_T/dt_R`, which comes out of the light-time solution and so is the
/// caller's — `1` for a stationary link, `1 − ρ̇/c` to first order for a receding one. It has no
/// default, because a silent `1` on a real deep-space link is a six-figure error in hertz.
#[pyfunction]
fn dsn_measured_frequency(
    segments: Vec<(f64, f64, f64, f64)>,
    uplink: &str,
    downlink: &str,
    transmit_epoch: f64,
    transmit_rate: f64,
) -> PyResult<f64> {
    let owned = ramp_segments(&segments);
    let table = RampTable::new(&owned).map_err(value_err)?;
    estimation_measured_frequency(
        &table,
        dsn_band(uplink)?,
        dsn_band(downlink)?,
        transmit_epoch,
        transmit_rate,
    )
    .map_err(value_err)
}

/// ΔDOR against a quasar, seconds — the arrival-time difference across the baseline, differenced
/// again against a radio source at infinity so the station clocks and both atmospheres cancel.
///
/// `right_ascension` and `declination` are the catalogue angles of the source, in radians, in the
/// same equatorial frame the states are expressed in.
#[pyfunction]
fn delta_dor(
    satellite: [f64; 6],
    emission: f64,
    first: ([f64; 3], [f64; 3], f64),
    second: ([f64; 3], [f64; 3], f64),
    right_ascension: f64,
    declination: f64,
) -> PyResult<f64> {
    let geometry = tdoa_geometry(emission, first, second)?;
    Ok(estimation_delta_dor(
        &satellite,
        &geometry,
        Quasar {
            right_ascension,
            declination,
        },
    ))
}

/// Solar-corona range correction along one leg, **metres**.
///
/// `±(40.308/f²)∫n_e ds` along the straight line between the two **heliocentric** positions, with the
/// electron density an inverse power series `Σ c_i (r/R☉)^-k_i`. `series` is
/// `[(coefficient, exponent), …]`; omitting it takes Aksim & Pavlov (2022), which is Tudat's
/// default. `group_delay` selects the sign: a ranging observable arrives late, a carrier-phase or
/// Doppler one early, by the same magnitude.
///
/// Raises `ValueError` if the line of sight passes within one solar radius of the Sun's centre —
/// there is no honest correction for a signal through the photosphere, and returning zero or NaN
/// would each be a different way of not saying so.
#[pyfunction]
#[pyo3(signature = (
    transmitter_from_sun, receiver_from_sun, frequency, group_delay = true, series = None,
))]
fn solar_corona_delay(
    transmitter_from_sun: [f64; 3],
    receiver_from_sun: [f64; 3],
    frequency: f64,
    group_delay: bool,
    series: Option<Vec<(f64, f64)>>,
) -> PyResult<f64> {
    let owned: Vec<CoronaTerm> = series.map_or_else(
        || AKSIM_PAVLOV_2022.to_vec(),
        |terms| {
            terms
                .into_iter()
                .map(|(coefficient, exponent)| CoronaTerm {
                    coefficient,
                    exponent,
                })
                .collect()
        },
    );
    let model = CoronaSeries::new(&owned).map_err(value_err)?;
    estimation_solar_corona_delay(
        &model,
        &transmitter_from_sun,
        &receiver_from_sun,
        frequency,
        if group_delay {
            PlasmaSignature::GroupDelay
        } else {
            PlasmaSignature::PhaseAdvance
        },
    )
    .map_err(value_err)
}

// ---------------------------------------------------------------------------------------------
// Orbital lifetime and re-entry (#310), and element-set generation (#311).
// ---------------------------------------------------------------------------------------------

/// An exponential atmosphere about the spherical radius, co-rotating rigidly about `ẑ`.
///
/// The one atmosphere this slice of the bridge offers, and named rather than implied: `ρ₀ e^(−(r −
/// r₀)/H)` is a first-order decay model, not NRLMSISE-00. Driving the DSST quadrature with an
/// empirical model over a decade needs the inertial→geodetic rotation carried through every node,
/// which is a different piece of work and is recorded as one; quoting a decade of decay from a
/// model whose geographic arguments were quietly approximated would be worse than not offering it.
///
/// `scale` multiplies the density and is how an ensemble member differs from its neighbours: over a
/// decade the answer is dominated by how active the Sun turns out to be, and to first order that
/// enters as a factor on the density the object flies through.
struct DecayExponential {
    reference_density: f64,
    reference_radius: f64,
    scale_height: f64,
    omega: f64,
    scale: f64,
}

impl DecayAtmosphere for DecayExponential {
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

/// The decaying spacecraft, its zonal field, and the threshold it is declared re-entered at.
struct Decay {
    zonal: DsstZonal,
    drag: DsstAtmosphericDrag,
    mu: f64,
    body_radius: f64,
    reentry_altitude: f64,
}

impl Decay {
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

/// The three named outcomes as `(outcome, seconds, perigee_altitude)`.
///
/// A tagged tuple rather than a bare number, because the Rust side deliberately refuses to
/// collapse the three: a survivor has no decay epoch and an unresolved run has one it did not
/// observe, and presenting either as a very large or a very confident number is the mistake the
/// sum type exists to prevent. `"survives"` reports the horizon it outlived; `"unresolved"` reports
/// where the propagation stopped and the altitude reached, which is what tells a terminal plunge
/// from a broken force model.
fn lifetime_outcome(outcome: Lifetime) -> (&'static str, f64, f64) {
    match outcome {
        Lifetime::Reentry {
            seconds,
            perigee_altitude,
        } => ("reentry", seconds, perigee_altitude),
        Lifetime::Survives {
            horizon,
            perigee_altitude,
        } => ("survives", horizon, perigee_altitude),
        Lifetime::Unresolved {
            seconds,
            perigee_altitude,
        } => ("unresolved", seconds, perigee_altitude),
    }
}

/// Predict when one object re-enters, under an exponential atmosphere.
///
/// `initial` is the **mean** equinoctial state `[a, k, h, q, p, λ]` (metres and radians) and the
/// answer is seconds from that epoch. `zonals` are the unnormalised `Cₙ₀` indexed by degree, so
/// `[0, 0, C20]` is `J2` alone and `C20 = −J2`. The scan advances in `chunk` seconds and bisects
/// the first chunk across which the mean perigee altitude falls to `reentry_altitude`.
///
/// Returns `(outcome, seconds, perigee_altitude)` with `outcome` one of `"reentry"`, `"survives"`
/// or `"unresolved"`. Raises `ValueError` for a horizon, chunk or tolerance that is not positive and
/// finite, for a pair of them needing too many chunks, and for an initial state that is not
/// elliptic. A propagation that breaks down is **not** an error — it is `"unresolved"`, because it
/// is an outcome of the physics as often as of the arguments.
#[pyfunction]
#[pyo3(signature = (
    initial, zonals, cd, area_over_mass, reference_density, reference_radius, scale_height,
    horizon, chunk = 86_400.0, tolerance = 60.0, density_scale = 1.0, mu = 3.986_004_418e14,
    body_radius = 6_378_137.0, reentry_altitude = 120_000.0, rotation_rate = EARTH_ROTATION_RATE,
))]
#[allow(clippy::too_many_arguments)] // the state, the field, the spacecraft, the atmosphere, the scan
fn orbital_lifetime(
    initial: [f64; 6],
    zonals: Vec<f64>,
    cd: f64,
    area_over_mass: f64,
    reference_density: f64,
    reference_radius: f64,
    scale_height: f64,
    horizon: f64,
    chunk: f64,
    tolerance: f64,
    density_scale: f64,
    mu: f64,
    body_radius: f64,
    reentry_altitude: f64,
    rotation_rate: f64,
) -> PyResult<(&'static str, f64, f64)> {
    let decay = decay_model(
        zonals,
        cd,
        area_over_mass,
        mu,
        body_radius,
        reentry_altitude,
    )?;
    let atmosphere = DecayExponential {
        reference_density,
        reference_radius,
        scale_height,
        omega: rotation_rate,
        scale: density_scale,
    };
    estimation_lifetime(
        &decay.model(),
        &atmosphere,
        initial,
        horizon,
        chunk,
        tolerance,
    )
    .map(lifetime_outcome)
    .map_err(value_err)
}

/// The same decay under a set of density scale factors, reported as quantiles.
///
/// One member per entry of `density_scales`, which is where the driver uncertainty enters: over a
/// decade the answer is dominated by how active the Sun turns out to be, so a single number is the
/// median of a distribution nobody computed. Returns
/// `(quantile_seconds, members, survivors, unresolved)`, where `quantile_seconds[i]` is the
/// re-entry epoch at `quantiles[i]` or `None` when the ensemble has no members at all.
///
/// A non-zero `unresolved` is a finding rather than a statistic: those members stopped being
/// propagable before reaching the threshold, and every quantile reported is conditioned on that.
#[pyfunction]
#[pyo3(signature = (
    initial, zonals, cd, area_over_mass, reference_density, reference_radius, scale_height,
    horizon, density_scales, quantiles, chunk = 86_400.0, tolerance = 60.0,
    mu = 3.986_004_418e14, body_radius = 6_378_137.0, reentry_altitude = 120_000.0,
    rotation_rate = EARTH_ROTATION_RATE,
))]
#[allow(clippy::too_many_arguments)] // as `orbital_lifetime`, plus the ensemble and its quantiles
fn orbital_lifetime_ensemble(
    initial: [f64; 6],
    zonals: Vec<f64>,
    cd: f64,
    area_over_mass: f64,
    reference_density: f64,
    reference_radius: f64,
    scale_height: f64,
    horizon: f64,
    density_scales: Vec<f64>,
    quantiles: Vec<f64>,
    chunk: f64,
    tolerance: f64,
    mu: f64,
    body_radius: f64,
    reentry_altitude: f64,
    rotation_rate: f64,
) -> PyResult<(Vec<Option<f64>>, usize, usize, usize)> {
    let decay = decay_model(
        zonals,
        cd,
        area_over_mass,
        mu,
        body_radius,
        reentry_altitude,
    )?;
    let ensemble = estimation_ensemble_lifetime(
        &decay.model(),
        |member| DecayExponential {
            reference_density,
            reference_radius,
            scale_height,
            omega: rotation_rate,
            scale: density_scales[member],
        },
        density_scales.len(),
        initial,
        horizon,
        chunk,
        tolerance,
    )
    .map_err(|(member, error)| PyValueError::new_err(format!("member {member}: {error}")))?;
    Ok((
        quantiles
            .into_iter()
            .map(|p| ensemble.quantile(p))
            .collect(),
        ensemble.members(),
        ensemble.survivors(),
        ensemble.unresolved(),
    ))
}

/// `∂ln(lifetime) / ∂ln(S/m)` by central difference on the area-to-mass ratio.
///
/// The elasticity a disposal study needs: how much of a lifetime change a drag-augmentation device
/// buys, in the units the requirement is written in. `fraction` is the relative step taken either
/// side of `area_over_mass`. Raises `ValueError` if either side fails to re-enter within the
/// horizon, because a sensitivity between two epochs one of which does not exist is not a number.
#[pyfunction]
#[pyo3(signature = (
    initial, zonals, cd, area_over_mass, reference_density, reference_radius, scale_height,
    horizon, chunk = 86_400.0, tolerance = 60.0, fraction = 0.01, density_scale = 1.0,
    mu = 3.986_004_418e14, body_radius = 6_378_137.0, reentry_altitude = 120_000.0,
    rotation_rate = EARTH_ROTATION_RATE,
))]
#[allow(clippy::too_many_arguments)] // as `orbital_lifetime`, plus the difference step
fn ballistic_log_sensitivity(
    initial: [f64; 6],
    zonals: Vec<f64>,
    cd: f64,
    area_over_mass: f64,
    reference_density: f64,
    reference_radius: f64,
    scale_height: f64,
    horizon: f64,
    chunk: f64,
    tolerance: f64,
    fraction: f64,
    density_scale: f64,
    mu: f64,
    body_radius: f64,
    reentry_altitude: f64,
    rotation_rate: f64,
) -> PyResult<f64> {
    let decay = decay_model(
        zonals,
        cd,
        area_over_mass,
        mu,
        body_radius,
        reentry_altitude,
    )?;
    let atmosphere = DecayExponential {
        reference_density,
        reference_radius,
        scale_height,
        omega: rotation_rate,
        scale: density_scale,
    };
    estimation_ballistic_sensitivity(
        &decay.model(),
        &atmosphere,
        initial,
        horizon,
        chunk,
        tolerance,
        fraction,
    )
    .map_err(value_err)
}

/// The zonal field and the cannonball, validated once for all three lifetime entry points.
fn decay_model(
    zonals: Vec<f64>,
    cd: f64,
    area_over_mass: f64,
    mu: f64,
    body_radius: f64,
    reentry_altitude: f64,
) -> PyResult<Decay> {
    // `DsstZonal::new` asserts on a short list, and an assertion across the FFI boundary is an
    // abort rather than an exception -- so the length is checked here, where it can be said.
    if zonals.len() < 3 {
        return Err(PyValueError::new_err(
            "zonals must run to at least degree 2 (J2), i.e. `[0.0, 0.0, C20]`",
        ));
    }
    Ok(Decay {
        zonal: DsstZonal::new(mu, body_radius, zonals),
        drag: DsstAtmosphericDrag::new(cd, area_over_mass, mu),
        mu,
        body_radius,
        reentry_altitude,
    })
}

/// Print an element set as the two lines of a TLE.
///
/// `epoch` is `(year, month, day, hour, minute, second)` UTC; `parameters` are the six fitted
/// fields in the element set's own **printed** units and order — mean motion in revolutions per
/// day, eccentricity, then inclination, right ascension, argument of perigee and mean anomaly in
/// degrees. `drag_term` is `B*`, the only drag quantity SGP4 reads; the two mean-motion derivatives
/// are printed as zero because nothing here fits them and a plausible value would be indistinguishable
/// from a fitted one.
///
/// Returns `(line1, line2)`, 69 ASCII columns each with both checksums correct. Raises `ValueError`
/// for a catalog number past 99 999, an epoch outside 1957–2056, a malformed international
/// designator, and any field whose columns cannot express its value.
#[pyfunction]
#[pyo3(signature = (
    epoch, norad_id, parameters, drag_term = 0.0, international_designator = None,
    element_set_number = 0, revolution_number = 0,
))]
fn tle_lines(
    epoch: (i32, u8, u8, u8, u8, f64),
    norad_id: u32,
    parameters: [f64; 6],
    drag_term: f64,
    international_designator: Option<&str>,
    element_set_number: u64,
    revolution_number: u64,
) -> PyResult<(String, String)> {
    let (year, month, day, hour, minute, second) = epoch;
    let utc = UtcDateTime::new(year, month, day, hour, minute, second).map_err(value_err)?;
    let seeded =
        seed_elements(utc, NoradId::new(norad_id), &parameters).map_err(|e| value_err(&e))?;
    // Through `TleModel` rather than writing the fields back by hand: it is the same write-back the
    // fit uses, so a set printed here and a set fitted there differ in their values and not in
    // their conventions -- including the reduction of every angle into `[0, 360)`.
    let model = TleModel::new(seeded, Convention::Reference);
    let elements = model.elements_with_drag(&[
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[3],
        parameters[4],
        parameters[5],
        drag_term,
    ]);
    let mut builder = TleBuilder::from_elements(&elements)
        .element_set_number(element_set_number)
        .revolution_number(revolution_number);
    if let Some(designator) = international_designator {
        builder = builder.international_designator(designator);
    }
    let tle = builder.build().map_err(|e| value_err(&e))?;
    Ok((tle.line1().to_owned(), tle.line2().to_owned()))
}

// ---------------------------------------------------------------------------------------------
// The GNSS measurement side (#312): the observation file, the dual-frequency combinations formed
// from it, the cycle-slip detectors that cut its carrier-phase arcs, and the integer least-squares
// that resolves the ambiguities. The detectors are stateful in Rust and batch here — a Python
// caller feeding one sample at a time across the FFI boundary would pay the crossing per sample and
// gain nothing, since the arc is the unit the verdicts are read on.
// ---------------------------------------------------------------------------------------------

/// One observation slot: `(value, lli, signal_strength)`, or `None` where the file carries none.
type ObsSlot = Option<(f64, Option<u8>, Option<u8>)>;
/// One satellite's slots at one epoch, aligned with its constellation's declared descriptors.
type ObsRecord = (String, Vec<ObsSlot>);
/// `(seconds_j2000, power_failure, receiver_clock_offset, satellites)`.
type ObsEpochTuple = (f64, bool, Option<f64>, Vec<ObsRecord>);
/// `(version, file_system, marker_name, approx_position, antenna_delta_hen, interval, time_system,
/// obs_types)`.
type ObsHeaderTuple = (
    f64,
    char,
    String,
    Option<[f64; 3]>,
    Option<[f64; 3]>,
    Option<f64>,
    &'static str,
    Vec<(char, Vec<String>)>,
);

fn time_system_label(system: TimeSystem) -> &'static str {
    match system {
        TimeSystem::Gps => "GPS",
        TimeSystem::Glo => "GLO",
        TimeSystem::Gal => "GAL",
        TimeSystem::Tai => "TAI",
        TimeSystem::Utc => "UTC",
        TimeSystem::Qzs => "QZS",
        TimeSystem::Bdt => "BDT",
        TimeSystem::Irn => "IRN",
    }
}

fn header_tuple(header: &ObsHeader) -> ObsHeaderTuple {
    (
        header.version,
        header.file_system,
        header.marker_name.clone(),
        header.approx_position,
        header.antenna_delta_hen,
        header.interval,
        time_system_label(header.time_system),
        header
            .obs_types
            .iter()
            .map(|(system, codes)| {
                (
                    *system,
                    codes
                        .iter()
                        .map(|c| c.to_string().trim_end().to_owned())
                        .collect(),
                )
            })
            .collect(),
    )
}

fn split_sat_id(sat: &str) -> PyResult<(char, u8)> {
    let reject =
        || PyValueError::new_err(format!("a satellite is a letter and a PRN, not {sat:?}"));
    let system = sat.chars().next().ok_or_else(reject)?;
    if !system.is_ascii_alphabetic() {
        return Err(reject());
    }
    let prn = sat[system.len_utf8()..]
        .trim()
        .parse::<u8>()
        .map_err(|_| reject())?;
    Ok((system, prn))
}

/// The position of `code` in the declared descriptor list for `system`, which is the slot every data
/// record for that constellation reports it in. Matched on the descriptor's own written form rather
/// than on a parse of the caller's string, so the header's list stays the single authority.
fn descriptor_slot(header: &ObsHeader, system: char, code: &str) -> Option<usize> {
    header
        .obs_types
        .get(&system)?
        .iter()
        .position(|c| c.to_string().trim_end() == code)
}

/// Read a RINEX 3 observation file, returning `(header, epochs)`.
///
/// The header is `(version, file_system, marker_name, approx_position, antenna_delta_hen, interval,
/// time_system, obs_types)`, and `obs_types` is `[(constellation, [descriptor])]` in the order every
/// data record for that constellation reports its values. That order is the whole alignment: each
/// epoch is `(seconds_j2000, power_failure, receiver_clock_offset, [(satellite, slots)])` where
/// `slots` holds one `(value, lli, signal_strength)` per declared descriptor, `None` where the file
/// carries no observation. Raises `ValueError` naming the line at fault.
#[pyfunction]
fn rinex_obs(text: &str) -> PyResult<(ObsHeaderTuple, Vec<ObsEpochTuple>)> {
    let file = RinexObs::parse(text).map_err(value_err)?;
    let epochs = file
        .epochs
        .iter()
        .map(|epoch| {
            let satellites = epoch
                .satellites
                .iter()
                .map(|s| {
                    let values = s
                        .values
                        .iter()
                        .map(|v| v.map(|o| (o.value, o.lli, o.signal_strength)))
                        .collect();
                    (format!("{}{:02}", s.sat.system, s.sat.prn), values)
                })
                .collect();
            (
                epoch.seconds_j2000,
                epoch.power_failure,
                epoch.receiver_clock_offset,
                satellites,
            )
        })
        .collect();
    Ok((header_tuple(&file.header), epochs))
}

/// One satellite's series of one observation descriptor: `[(seconds_j2000, value, lost_lock)]`,
/// skipping the epochs the satellite is absent from or carries no value at.
///
/// `sat` is a RINEX identifier such as `"G01"` and `code` a descriptor such as `"C1C"`. `lost_lock`
/// is the receiver's own loss-of-lock bit — a slip it declares, independent of anything
/// [`detect_cycle_slips_geometry_free`] concludes. Raises `ValueError` if the header declares no
/// such descriptor for that constellation, which is the difference between "not observed" and
/// "asked for the wrong thing".
#[pyfunction]
fn rinex_obs_series(text: &str, sat: &str, code: &str) -> PyResult<Vec<(f64, f64, bool)>> {
    let (system, prn) = split_sat_id(sat)?;
    let file = RinexObs::parse(text).map_err(value_err)?;
    let slot = descriptor_slot(&file.header, system, code).ok_or_else(|| {
        PyValueError::new_err(format!(
            "the header declares no {code:?} for constellation {system:?}"
        ))
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
            Some((
                epoch.seconds_j2000,
                observation.value,
                observation.lost_lock(),
            ))
        })
        .collect())
}

/// The carrier frequency in Hz of `band` on constellation `system`, or `None` where that
/// constellation does not broadcast it. GLONASS L1/L2 are FDMA and answer `None` — their frequency
/// depends on the satellite's channel, which [`glonass_l1_frequency`] takes.
#[pyfunction]
fn carrier_frequency(system: char, band: char) -> Option<f64> {
    gnss_carrier_frequency(system, band)
}

fn frequency_pair(f1: f64, f2: f64) -> PyResult<FrequencyPair> {
    FrequencyPair::new(f1, f2).map_err(value_err)
}

/// The four wavelengths of a frequency pair, metres: `(lambda1, lambda2, wide_lane, narrow_lane)`.
///
/// The wide lane is `c/(f1 - f2)` and the narrow lane `c/(f1 + f2)` — 0.861 m and 0.107 m on GPS
/// L1/L2, which is why the wide-lane ambiguity is the one resolved first.
#[pyfunction]
fn lane_wavelengths(f1: f64, f2: f64) -> PyResult<(f64, f64, f64, f64)> {
    let pair = frequency_pair(f1, f2)?;
    Ok((
        pair.wavelength1(),
        pair.wavelength2(),
        pair.wide_lane_wavelength(),
        pair.narrow_lane_wavelength(),
    ))
}

/// Ionosphere-free combination `(f1^2*x1 - f2^2*x2)/(f1^2 - f2^2)`, metres. Applies to a code pair or
/// a phase pair alike; on phases the surviving ambiguity is not an integer.
#[pyfunction]
fn ionosphere_free(f1: f64, f2: f64, x1: f64, x2: f64) -> PyResult<f64> {
    Ok(frequency_pair(f1, f2)?.ionosphere_free(x1, x2))
}

/// Wide-lane combination `(f1*x1 - f2*x2)/(f1 - f2)`, metres.
#[pyfunction]
fn wide_lane(f1: f64, f2: f64, x1: f64, x2: f64) -> PyResult<f64> {
    Ok(frequency_pair(f1, f2)?.wide_lane(x1, x2))
}

/// Narrow-lane combination `(f1*x1 + f2*x2)/(f1 + f2)`, metres.
#[pyfunction]
fn narrow_lane(f1: f64, f2: f64, x1: f64, x2: f64) -> PyResult<f64> {
    Ok(frequency_pair(f1, f2)?.narrow_lane(x1, x2))
}

/// Melbourne-Wübbena combination: the wide-lane phase less the narrow-lane code, from pseudoranges
/// `p1`/`p2` and carrier phases `l1`/`l2`, all metres.
///
/// In metres by default, and in wide-lane cycles with `cycles=True` — the unit the arc is read in,
/// since over a clean arc the value hovers about an integer and a step in it is a wide-lane slip.
#[pyfunction]
#[pyo3(signature = (f1, f2, p1, p2, l1, l2, cycles = false))]
#[allow(clippy::too_many_arguments)]
fn melbourne_wubbena(
    f1: f64,
    f2: f64,
    p1: f64,
    p2: f64,
    l1: f64,
    l2: f64,
    cycles: bool,
) -> PyResult<f64> {
    let pair = frequency_pair(f1, f2)?;
    Ok(if cycles {
        pair.melbourne_wubbena_cycles(p1, p2, l1, l2)
    } else {
        pair.melbourne_wubbena(p1, p2, l1, l2)
    })
}

/// Geometry-free combination `x1 - x2`, metres — everything non-dispersive cancels, which is why it
/// takes no frequencies.
#[pyfunction]
fn geometry_free(x1: f64, x2: f64) -> f64 {
    gnss_geometry_free(x1, x2)
}

/// GRAPHIC combination `(p + l)/2`, metres — the single-frequency ionosphere-free observable.
#[pyfunction]
fn graphic(p: f64, l: f64) -> f64 {
    gnss_graphic(p, l)
}

fn slip_verdict(verdict: SlipVerdict) -> (&'static str, Option<f64>) {
    match verdict {
        SlipVerdict::ArcStart => ("arc_start", None),
        SlipVerdict::Continuous { statistic } => ("continuous", Some(statistic)),
        SlipVerdict::Slipped { statistic } => ("slipped", Some(statistic)),
    }
}

fn slip_series(
    times: &[f64],
    values: &[f64],
    mut update: impl FnMut(f64, f64) -> SlipVerdict,
) -> PyResult<Vec<(&'static str, Option<f64>)>> {
    if times.len() != values.len() {
        return Err(PyValueError::new_err(format!(
            "{} times do not go with {} values",
            times.len(),
            values.len()
        )));
    }
    Ok(times
        .iter()
        .zip(values)
        .map(|(&t, &v)| slip_verdict(update(t, v)))
        .collect())
}

/// Run the first-difference cycle-slip detector over a test series, returning one verdict per
/// sample: `("arc_start", None)`, `("continuous", statistic)` or `("slipped", statistic)`.
///
/// `threshold` is on the raw first difference in the units of `values`, so it has to be chosen for
/// the sampling interval — on the geometry-free combination at 30 s, a few centimetres separates
/// real ionospheric change from the 0.05-0.11 m a one-cycle L1 or L2 slip moves it by. A gap longer
/// than `max_gap` seconds restarts the arc rather than testing across it.
#[pyfunction]
fn detect_cycle_slips_geometry_free(
    times: Vec<f64>,
    values: Vec<f64>,
    threshold: f64,
    max_gap: f64,
) -> PyResult<Vec<(&'static str, Option<f64>)>> {
    let mut detector = GeometryFreeDetector::new(threshold, max_gap).map_err(value_err)?;
    slip_series(&times, &values, |t, v| detector.update(t, v))
}

/// Run the polynomial-fit cycle-slip detector over a test series, returning the verdicts of
/// [`detect_cycle_slips_geometry_free`].
///
/// The fit is over the last `window` samples at degree `order`, and the statistic is the residual of
/// the next sample against the prediction — which is what sees a slip smaller than the series' own
/// drift over the window. The first `window` samples are `"arc_start"`: nothing is tested until the
/// window fills.
#[pyfunction]
fn detect_cycle_slips_polynomial(
    times: Vec<f64>,
    values: Vec<f64>,
    order: usize,
    window: usize,
    threshold: f64,
    max_gap: f64,
) -> PyResult<Vec<(&'static str, Option<f64>)>> {
    let mut detector =
        PolynomialFitDetector::new(order, window, threshold, max_gap).map_err(value_err)?;
    slip_series(&times, &values, |t, v| detector.update(t, v))
}

fn reduction(float: &[f64], covariance: &[f64]) -> PyResult<Reduction> {
    Reduction::new(float, covariance).map_err(value_err)
}

/// Decorrelate an ambiguity problem, returning `(transformation, conditional_variances)`.
///
/// `covariance` is the row-major `n x n` covariance and `float_ambiguities` the `n` float values.
/// The transformation is the integer-valued `Z` of LAMBDA, row-major, and the conditional variances
/// are the `d` of `Q_z = L^T D L` — the product of which is the volume of the search ellipsoid, so
/// their spread is the measure of how much decorrelation bought.
#[pyfunction]
fn ambiguity_decorrelation(
    float_ambiguities: Vec<f64>,
    covariance: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let reduced = reduction(&float_ambiguities, &covariance)?;
    Ok((
        reduced.transformation().to_vec(),
        reduced.conditional_variances().to_vec(),
    ))
}

/// The bootstrapped integer solution and its objective: `(ambiguities, squared_norm)`.
///
/// Sequential conditional rounding in the decorrelated parametrisation — one pass, no search, and
/// not in general the integer least-squares solution. It is what bounds the search, and its success
/// rate is the classical lower bound on the ILS success rate.
#[pyfunction]
fn ambiguity_bootstrap(
    float_ambiguities: Vec<f64>,
    covariance: Vec<f64>,
) -> PyResult<(Vec<i64>, f64)> {
    let candidate = reduction(&float_ambiguities, &covariance)?.bootstrap();
    Ok((candidate.ambiguities, candidate.squared_norm))
}

/// The `count` best integer ambiguity vectors by the LAMBDA integer least-squares search:
/// `[(ambiguities, squared_norm)]`, best first, in the caller's original parametrisation.
///
/// The ratio of the second squared norm to the first is the classical acceptance test. Raises
/// `ValueError` if the covariance is not a covariance, or if the enumeration reaches its node budget
/// without exhausting the ellipsoid — a truncated list would be a wrong answer that looks like a
/// right one.
#[pyfunction]
#[pyo3(signature = (float_ambiguities, covariance, count = 2))]
fn resolve_ambiguities(
    float_ambiguities: Vec<f64>,
    covariance: Vec<f64>,
    count: usize,
) -> PyResult<Vec<(Vec<i64>, f64)>> {
    Ok(reduction(&float_ambiguities, &covariance)?
        .search(count)
        .map_err(value_err)?
        .into_iter()
        .map(|c| (c.ambiguities, c.squared_norm))
        .collect())
}

// ---------------------------------------------------------------------------------------------
// Conjunction assessment (#92, #315, #322): the screen that finds a TCA, the 2-D collision
// probability at one, and the four checks that say whether that probability is inside its own
// model's domain. The two scans take a Python callable for the relative state, on the precedent
// `gauss_newton` set: the first exception it raises is held and re-raised here, and the sample it
// could not produce crosses as NaN so the Rust scan stops where the ephemeris did rather than
// running to the end of a span it has no data for.
// ---------------------------------------------------------------------------------------------

/// A Python callable standing in for a Rust sampler, holding the first exception it raised.
struct PySampler<'py> {
    func: &'py Bound<'py, PyAny>,
    err: core::cell::RefCell<Option<PyErr>>,
}

impl<'py> PySampler<'py> {
    fn new(func: &'py Bound<'py, PyAny>) -> Self {
        Self {
            func,
            err: core::cell::RefCell::new(None),
        }
    }

    /// Call it, or record the failure and answer with the non-finite sample that ends the scan.
    fn call<T>(&self, t: f64, unusable: T) -> T
    where
        T: for<'a, 'p> FromPyObject<'a, 'p, Error = PyErr>,
    {
        if self.err.borrow().is_some() {
            return unusable;
        }
        match self.func.call1((t,)).and_then(|r| r.extract::<T>()) {
            Ok(sample) => sample,
            Err(e) => {
                *self.err.borrow_mut() = Some(e);
                unusable
            }
        }
    }
}

fn conjunction_geometry(
    relative_position: [f64; 3],
    relative_velocity: [f64; 3],
    primary_covariance: [[f64; 3]; 3],
    secondary_covariance: [[f64; 3]; 3],
    primary_radius: f64,
    secondary_radius: f64,
) -> ConjunctionGeometry {
    ConjunctionGeometry::new(
        relative_position,
        relative_velocity,
        primary_covariance,
        secondary_covariance,
        primary_radius,
        secondary_radius,
    )
}

/// The 2-D collision probability at one conjunction, by all four estimators:
/// `(foster, patera, chan, maximum)`.
///
/// The state is relative — `r2 - r1` and `v2 - v1` at TCA, metres and m/s in one common inertial
/// frame — and each covariance is the 3x3 position covariance of that object, m², in the same frame.
/// The radii are hard-body equivalent-sphere radii, metres, and their sum is the collision cross
/// section.
///
/// All four are returned rather than one chosen here: Foster, Patera and Chan integrate the same
/// encounter-plane Gaussian by different routes and agree to a few percent where the 2-D model holds,
/// so their spread is itself the diagnostic. The fourth is the *maximum* probability over a scaling
/// of the covariance — a bound rather than an estimate, and the one to act on when the covariance is
/// not trusted. [`conjunction_usage_violations`] is what says whether any of them applies at all.
#[pyfunction]
fn collision_probability(
    relative_position: [f64; 3],
    relative_velocity: [f64; 3],
    primary_covariance: [[f64; 3]; 3],
    secondary_covariance: [[f64; 3]; 3],
    primary_radius: f64,
    secondary_radius: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    let plane = conjunction_geometry(
        relative_position,
        relative_velocity,
        primary_covariance,
        secondary_covariance,
        primary_radius,
        secondary_radius,
    )
    .project()
    .map_err(value_err)?;
    Ok((
        plane.probability_foster(),
        plane.probability_patera(),
        plane.probability_chan(),
        plane.probability_max(),
    ))
}

/// Whether the 2-D collision probability is inside its own model's domain, returning
/// `((any, non_positive_definite, offset, extended, inaccuracy), (offset_indicator,
/// extended_indicator, inaccuracy_indicator), correction_factor)`.
///
/// The geometry is [`collision_probability`]'s. `minimum_period` is the shorter of the two objects'
/// orbital periods in seconds — the scale that turns an offset and a duration into the dimensionless
/// ratios the published thresholds are stated on, and there is no default for it because a guessed
/// period rescales both thresholds silently. `sample(t)` returns `((position, velocity), covariance)`
/// — the relative state and the *joint* covariance `C1 + C2`, `t` seconds from TCA, under whatever
/// dynamics the caller propagates with.
///
/// A flag says the 2-D estimate is not known to be accurate, not that it is wrong; the thresholds are
/// deliberately conservative. `correction_factor` is the rough multiplicative correction that falls
/// out of the fourth check: `Pc ~ correction_factor * Pc_2D`.
#[pyfunction]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn conjunction_usage_violations(
    relative_position: [f64; 3],
    relative_velocity: [f64; 3],
    primary_covariance: [[f64; 3]; 3],
    secondary_covariance: [[f64; 3]; 3],
    primary_radius: f64,
    secondary_radius: f64,
    minimum_period: f64,
    sample: &Bound<'_, PyAny>,
) -> PyResult<((bool, bool, bool, bool, bool), (f64, f64, f64), f64)> {
    let geometry = conjunction_geometry(
        relative_position,
        relative_velocity,
        primary_covariance,
        secondary_covariance,
        primary_radius,
        secondary_radius,
    );
    let sampler = PySampler::new(sample);
    let unusable = (([f64::NAN; 3], [f64::NAN; 3]), [[f64::NAN; 3]; 3]);
    let assessed = estimation_usage_violations(&geometry, minimum_period, |t| {
        let ((position, velocity), covariance) = sampler.call(t, unusable);
        RelativeSample {
            position,
            velocity,
            covariance,
        }
    });
    if let Some(e) = sampler.err.into_inner() {
        return Err(e);
    }
    let violations = assessed.map_err(value_err)?;
    Ok((
        (
            violations.any(),
            violations.non_positive_definite(),
            violations.offset(),
            violations.extended(),
            violations.inaccuracy(),
        ),
        (
            violations.offset_indicator(),
            violations.extended_indicator(),
            violations.inaccuracy_indicator(),
        ),
        violations.correction_factor(),
    ))
}

/// Every closest approach of a relative trajectory in `[start, end]`, in time order:
/// `[(tca, distance, relative_speed)]`, metres and m/s.
///
/// `relative(t)` returns `(position, velocity)` — object 2 minus object 1, in one common inertial
/// frame, the convention [`collision_probability`] takes. The scan samples every `step` seconds and
/// refines each bracket by bisection to `tolerance`, so the cost is `ceil(log2(step/tolerance))`
/// evaluations per approach and does not depend on how the separation behaves inside the bracket.
///
/// The step is the resolution: two minima closer together than one step are reported as one, so the
/// step belongs below the shortest interval between successive extrema of the separation. Raises
/// `ValueError` for a step, tolerance or span that is not usable, and whatever `relative` raises —
/// a sample it cannot produce ends the scan at that epoch rather than inventing approaches from
/// comparisons against a missing state.
#[pyfunction]
fn closest_approaches(
    relative: &Bound<'_, PyAny>,
    start: f64,
    end: f64,
    step: f64,
    tolerance: f64,
) -> PyResult<Vec<(f64, f64, f64)>> {
    let sampler = PySampler::new(relative);
    let unusable = ([f64::NAN; 3], [f64::NAN; 3]);
    let mut found = Vec::new();
    let scanned = estimation_closest_approaches(
        |t| sampler.call(t, unusable),
        start,
        end,
        step,
        tolerance,
        &mut |approach| found.push((approach.tca, approach.distance, approach.relative_speed)),
    );
    if let Some(e) = sampler.err.into_inner() {
        return Err(e);
    }
    scanned.map_err(value_err)?;
    Ok(found)
}

/// Whether two orbits are far enough apart in radius that no conjunction is possible — the first and
/// cheapest stage of an all-on-all screen (Hoots, Crawford & Roehrich 1984, filter 1).
///
/// All five arguments are metres: the two orbits' geocentric perigee and apogee distances, and the
/// screening distance below which a miss is worth propagating. `True` is a geometric certainty and
/// not a heuristic — the two spherical shells the orbits are confined to cannot come within
/// `threshold` of each other at any epoch, whatever the phasing or the plane. The converse is not
/// claimed: most pairs that survive the filter still do not conjunct. A non-finite input answers
/// `False`, so a corrupt element set is examined rather than silently discarded.
#[pyfunction]
fn radially_separated(
    perigee_a: f64,
    apogee_a: f64,
    perigee_b: f64,
    apogee_b: f64,
    threshold: f64,
) -> bool {
    estimation_radially_separated(perigee_a, apogee_a, perigee_b, apogee_b, threshold)
}

/// The Shapiro (gravitational) delay on one leg, metres.
///
/// `r_emitter` and `r_receiver` are the two endpoints' geocentric distances and `path` the
/// straight-line distance between them, all metres; `gm` is the central body's gravitational
/// parameter, m³/s². There is no default for `gm` — a wrong one rescales the answer silently.
///
/// One leg, at Earth: 0.669 mm to a 500 km satellite at the zenith, 9.516 mm to LAGEOS, 12.66 mm to
/// GPS. A two-way system doubles those in flight time, but a two-way range published as a one-way
/// range carries one leg.
#[pyfunction]
fn shapiro_delay(r_emitter: f64, r_receiver: f64, path: f64, gm: f64) -> f64 {
    estimation_shapiro_delay(r_emitter, r_receiver, path, gm)
}

// ── Control: the loop, its latency, and the two bounds the pair of them produce ────────────────

/// The history capacity [`attitude_loop_run`] gives the loop, in samples.
///
/// `History<12, C>` stores its ring inline, so this is `C · 12 · 8` bytes of stack — 49 KB here.
/// The Rust API makes the capacity a const parameter precisely so a caller can size it; a Python
/// caller has no const parameter to give, so the binding picks one and refuses a lag that does not
/// fit rather than wrapping the ring around and reading a state from a previous era.
const LOOP_HISTORY: usize = 512;

/// Either attitude law, selected by name because a Python caller has no type to dispatch on.
///
/// Both laws implement [`ControlLaw`] and [`SmallAngleGains`], so carrying the choice in one enum
/// keeps every control entry point below on the same `(law, proportional, derivative)` triple: the
/// two are interchangeable in the bindings exactly where they are interchangeable in the crate.
#[derive(Clone, Copy)]
enum AttitudeLawKind {
    QuaternionPd(QuaternionPd),
    MrpLyapunov(MrpLyapunov),
}

impl AttitudeLawKind {
    fn build(law: &str, proportional: f64, derivative: f64) -> PyResult<Self> {
        let built = match law {
            "quaternion_pd" => QuaternionPd::new(proportional, derivative).map(Self::QuaternionPd),
            "mrp_lyapunov" => MrpLyapunov::new(proportional, derivative).map(Self::MrpLyapunov),
            other => {
                return Err(PyValueError::new_err(format!(
                    "law must be 'quaternion_pd' or 'mrp_lyapunov', not {other:?}"
                )));
            }
        };
        built.ok_or_else(|| {
            PyValueError::new_err(format!(
                "both gains must be finite and strictly positive, not ({proportional}, {derivative})"
            ))
        })
    }
}

impl ControlLaw for AttitudeLawKind {
    fn torque(&self, error: &AttitudeError, inertia: &Inertia) -> [f64; 3] {
        match self {
            Self::QuaternionPd(law) => law.torque(error, inertia),
            Self::MrpLyapunov(law) => law.torque(error, inertia),
        }
    }
}

impl SmallAngleGains for AttitudeLawKind {
    fn small_angle_gains(&self) -> (f64, f64) {
        match self {
            Self::QuaternionPd(law) => law.small_angle_gains(),
            Self::MrpLyapunov(law) => law.small_angle_gains(),
        }
    }
}

/// A row-major 3×3 from nine numbers, the layout every rotation and inertia argument below uses.
const fn mat3_of(flat: [f64; 9]) -> [[f64; 3]; 3] {
    [
        [flat[0], flat[1], flat[2]],
        [flat[3], flat[4], flat[5]],
        [flat[6], flat[7], flat[8]],
    ]
}

fn inertia_of(flat: [f64; 9]) -> PyResult<Inertia> {
    Inertia::from_tensor(mat3_of(flat)).ok_or_else(|| {
        PyValueError::new_err("the inertia tensor must be symmetric and positive definite")
    })
}

/// A `ROWS × COLS` matrix from a flat row-major list, or the length it wanted.
fn rows_of<const COLS: usize, const ROWS: usize>(flat: &[f64]) -> PyResult<[[f64; COLS]; ROWS]> {
    if flat.len() != ROWS * COLS {
        return Err(PyValueError::new_err(format!(
            "expected {} entries for a {ROWS}×{COLS} matrix, got {}",
            ROWS * COLS,
            flat.len()
        )));
    }
    Ok(core::array::from_fn(|i| {
        core::array::from_fn(|j| flat[i * COLS + j])
    }))
}

fn vector_of<const N: usize>(flat: &[f64]) -> PyResult<[f64; N]> {
    if flat.len() != N {
        return Err(PyValueError::new_err(format!(
            "expected {N} entries, got {}",
            flat.len()
        )));
    }
    Ok(core::array::from_fn(|i| flat[i]))
}

fn flatten<const COLS: usize, const ROWS: usize>(m: &[[f64; COLS]; ROWS]) -> Vec<f64> {
    m.iter().flat_map(|row| row.iter().copied()).collect()
}

/// The total latency in seconds, `τ_s + τ_c + τ_a`.
///
/// The three are separate arguments and not one `τ` because they have different physical origins
/// and different fixes — a star tracker's integration time is bought down with aperture, compute
/// latency with a faster processor, actuator lag with a different wheel. `ValueError` unless all
/// three are finite and non-negative: a negative latency is a loop acting on a measurement it has
/// not taken yet, and there is no sensible clamp for that.
#[pyfunction]
fn latency_total(sensor: f64, compute: f64, actuator: f64) -> PyResult<f64> {
    Ok(latency_of(sensor, compute, actuator)?.total())
}

fn latency_of(sensor: f64, compute: f64, actuator: f64) -> PyResult<Latency> {
    Latency::new(sensor, compute, actuator).ok_or_else(|| {
        PyValueError::new_err("each latency must be finite and non-negative, in seconds")
    })
}

/// How many whole steps of `step` the total latency is.
///
/// `ValueError` when it is not a whole number of them to within four ulps. Refusing is the point: a
/// loop whose delay is 2.5 steps is a different problem from one whose delay is 2 or 3 — it needs
/// the dense output of the step the delay falls inside — and rounding here would answer a question
/// nobody asked while labelling it with the caller's τ.
#[pyfunction]
fn latency_steps(sensor: f64, compute: f64, actuator: f64, step: f64) -> PyResult<usize> {
    latency_of(sensor, compute, actuator)?
        .steps(step)
        .ok_or_else(|| {
            PyValueError::new_err("the total latency is not a whole number of steps of `step`")
        })
}

/// The attitude error between `rotation` and `reference`, as `(mrp, angle)`.
///
/// Both matrices are nine numbers, row-major, inertial-from-body `[NB]` (body vectors to inertial;
/// the transpose of an attitude law's inertial→body matrix), and the error is `[NR]ᵀ[NB]`. `mrp` is the modified Rodrigues
/// vector of the error rotation and `angle` its principal angle in radians. The rates are body
/// rates in body axes, rad/s, and enter through the reference frame the error is measured in.
#[pyfunction]
fn attitude_error_mrp(
    rotation: [f64; 9],
    reference: [f64; 9],
    rate: [f64; 3],
    rate_reference: [f64; 3],
) -> ([f64; 3], f64) {
    let error = AttitudeError::new(
        &mat3_of(rotation),
        &mat3_of(reference),
        rate,
        rate_reference,
    );
    (error.mrp(), error.angle())
}

/// The same error as the error quaternion `(q₀, q₁, q₂, q₃)`, with `q₀ ≥ 0`.
///
/// Separate from [`attitude_error_mrp`] rather than returned alongside it because the two are the
/// same rotation in different charts and a caller wants one of them: the MRP is what the Lyapunov
/// law acts on, the quaternion is what most flight software speaks. The vector part `(q₁, q₂, q₃)`
/// is what a proportional law acts on.
#[pyfunction]
fn attitude_error_quaternion(
    rotation: [f64; 9],
    reference: [f64; 9],
    rate: [f64; 3],
    rate_reference: [f64; 3],
) -> [f64; 4] {
    AttitudeError::new(
        &mat3_of(rotation),
        &mat3_of(reference),
        rate,
        rate_reference,
    )
    .quaternion()
}

/// The commanded body moment, N·m, from either law.
///
/// `law` is `'quaternion_pd'` (Wie's `u = −K q_v − P δω`) or `'mrp_lyapunov'` (Schaub & Junkins,
/// which adds the gyroscopic feed-forward and is globally rather than locally stable). `inertia` is
/// the nine row-major entries of the tensor, kg·m².
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn attitude_torque(
    law: &str,
    proportional: f64,
    derivative: f64,
    rotation: [f64; 9],
    reference: [f64; 9],
    rate: [f64; 3],
    rate_reference: [f64; 3],
    inertia: [f64; 9],
) -> PyResult<[f64; 3]> {
    let law = AttitudeLawKind::build(law, proportional, derivative)?;
    let inertia = inertia_of(inertia)?;
    let error = AttitudeError::new(
        &mat3_of(rotation),
        &mat3_of(reference),
        rate,
        rate_reference,
    );
    Ok(law.torque(&error, &inertia))
}

/// `V = ½ δωᵀ I δω + 2K ln(1 + σᵀσ)` — the Lyapunov function the MRP law drives downhill.
///
/// Only the MRP law has one, so there is no `law` argument: the quaternion PD law is stable in the
/// small-angle linearisation and this function is not its certificate.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn attitude_lyapunov(
    proportional: f64,
    derivative: f64,
    rotation: [f64; 9],
    reference: [f64; 9],
    rate: [f64; 3],
    rate_reference: [f64; 3],
    inertia: [f64; 9],
) -> PyResult<f64> {
    let law = MrpLyapunov::new(proportional, derivative)
        .ok_or_else(|| PyValueError::new_err("both gains must be finite and strictly positive"))?;
    let inertia = inertia_of(inertia)?;
    let error = AttitudeError::new(
        &mat3_of(rotation),
        &mat3_of(reference),
        rate,
        rate_reference,
    );
    Ok(law.lyapunov(&error, &inertia))
}

/// `(k_θ, k_ω)` in N·m/rad and N·m·s/rad — the law's linearisation about the reference attitude.
///
/// The MRP law linearises to a *quarter* of its nominal stiffness, because `σ = q_v/(1+q₀) → δθ/4`,
/// so the two laws at the same nominal gains are not the same loop and their margins differ.
#[pyfunction]
fn attitude_small_angle_gains(
    law: &str,
    proportional: f64,
    derivative: f64,
) -> PyResult<(f64, f64)> {
    Ok(AttitudeLawKind::build(law, proportional, derivative)?.small_angle_gains())
}

/// The derivative gain that puts a principal axis of moment `moment` at ζ = 1: `P = √(2 I K)`.
///
/// Recovered from the law's own accessors as `ζ · K / ω_n` — since `ω_n = √(K/2I)` makes
/// `K/ω_n = √(2IK)` — rather than by writing `√(2 I K)` a second time here, where it could drift
/// from the crate's.
#[pyfunction]
fn quaternion_pd_critically_damped(proportional: f64, moment: f64) -> PyResult<f64> {
    let law = QuaternionPd::critically_damped(proportional, moment).ok_or_else(|| {
        PyValueError::new_err("the gain and the moment must be finite and strictly positive")
    })?;
    Ok(law.damping_ratio(moment) * proportional / law.natural_frequency(moment))
}

/// `(ω_n, ζ)` for the quaternion PD law on a principal axis of moment `moment`, rad/s and
/// dimensionless.
///
/// The moment is checked here rather than left to `√(K/2J)`: a non-positive one produces a NaN pair
/// that carries no clue where it came from, and `sl_quaternion_pd_response` refuses it.
#[pyfunction]
fn quaternion_pd_response(proportional: f64, derivative: f64, moment: f64) -> PyResult<(f64, f64)> {
    let law = QuaternionPd::new(proportional, derivative)
        .ok_or_else(|| PyValueError::new_err("both gains must be finite and strictly positive"))?;
    if !(moment.is_finite() && moment > 0.0) {
        return Err(PyValueError::new_err(
            "the moment must be finite and strictly positive",
        ));
    }
    Ok((law.natural_frequency(moment), law.damping_ratio(moment)))
}

/// What [`attitude_loop_run`] hands back: the sample times, the principal error angle at each, the
/// commanded moments flattened `3·steps`, the final rotation flattened row-major, and the final rate.
type LoopHistory = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, [f64; 3]);

/// Run a rigid-body attitude loop at a fixed rate under a fixed latency.
///
/// Returns `(times, angles, torques, final_rotation, final_rate)`: the sample times in seconds, the
/// principal error angle at each in radians, the commanded moment at each as a flat `3·steps` list
/// N·m, and the final state. The controller reads the state as it was `latency_steps` steps ago and
/// reads `seed_rotation`/`seed_rate` until that much time has elapsed.
///
/// `ValueError` unless the total latency is a whole number of `step`s and the resulting lag fits
/// the binding's history capacity of 512 samples.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn attitude_loop_run(
    py: Python<'_>,
    law: &str,
    proportional: f64,
    derivative: f64,
    sensor: f64,
    compute: f64,
    actuator: f64,
    step: f64,
    initial_rotation: [f64; 9],
    initial_rate: [f64; 3],
    seed_rotation: [f64; 9],
    seed_rate: [f64; 3],
    reference: [f64; 9],
    inertia: [f64; 9],
    steps: usize,
) -> PyResult<LoopHistory> {
    let law = AttitudeLawKind::build(law, proportional, derivative)?;
    let inertia = inertia_of(inertia)?;
    let latency = latency_of(sensor, compute, actuator)?;
    let closed = AttitudeLoop::new(law, latency, step).ok_or_else(|| {
        PyValueError::new_err(
            "the step must be finite and positive and the latency a whole number of steps of it",
        )
    })?;
    let initial = AttitudeState::new(mat3_of(initial_rotation), initial_rate);
    let seed = AttitudeState::new(mat3_of(seed_rotation), seed_rate);
    let reference = mat3_of(reference);

    let mut times = Vec::with_capacity(steps);
    let mut angles = Vec::with_capacity(steps);
    let mut torques = Vec::with_capacity(3 * steps);
    let final_state = py.detach(|| {
        closed.run::<LOOP_HISTORY>(initial, seed, &reference, &inertia, steps, |sample| {
            times.push(sample.time);
            angles.push(sample.error.angle());
            torques.extend_from_slice(&sample.torque);
        })
    });
    let final_state = final_state.ok_or_else(|| {
        PyValueError::new_err(format!(
            "a lag of {} steps does not fit the binding's history capacity of {LOOP_HISTORY}",
            closed.lag()
        ))
    })?;
    Ok((
        times,
        angles,
        torques,
        flatten(&final_state.rotation),
        final_state.rate,
    ))
}

/// The delay margin of one linearised principal axis.
///
/// Returns `(seconds, crossing_frequency, natural_frequency, damping_ratio, normalised)` — the
/// largest feedback delay the loop survives, the frequency the pole pair crosses at, the loop's own
/// `ω_n` and `ζ`, and `τ_max · ω_n`, which depends only on `ζ` and is the form to quote: it says the
/// margin is a fixed fraction of the loop's own period, so buying bandwidth spends margin
/// proportionally.
///
/// A zero rate gain is admitted, because the answer it produces — zero margin — is the right one.
#[pyfunction]
fn delay_margin_of_gains(
    moment: f64,
    angle_gain: f64,
    rate_gain: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let axis = SecondOrderLoop::new(moment, angle_gain, rate_gain).ok_or_else(|| {
        PyValueError::new_err(
            "the moment and the angle gain must be finite and positive and the rate gain non-negative",
        )
    })?;
    Ok(margin_tuple(&axis))
}

fn margin_tuple(axis: &SecondOrderLoop) -> (f64, f64, f64, f64, f64) {
    let margin = axis.delay_margin();
    (
        margin.seconds(),
        margin.crossing_frequency(),
        axis.natural_frequency(),
        axis.damping_ratio(),
        axis.normalised_delay_margin(),
    )
}

/// The delay margin of a named law on a principal axis of moment `moment`, in the shape
/// [`delay_margin_of_gains`] returns.
///
/// This is the composition to reach for rather than [`delay_margin_of_gains`] on hand-computed gains: it
/// takes the law's own linearisation, so the MRP law's quarter-stiffness is applied for you.
#[pyfunction]
fn delay_margin_of_law(
    law: &str,
    proportional: f64,
    derivative: f64,
    moment: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let law = AttitudeLawKind::build(law, proportional, derivative)?;
    let axis = SecondOrderLoop::of(&law, moment)
        .ok_or_else(|| PyValueError::new_err("the moment must be finite and strictly positive"))?;
    Ok(margin_tuple(&axis))
}

/// The binding axis of a three-axis body and its margin: `(axis, seconds, crossing_frequency)`.
///
/// `moments` are the three *principal* moments, kg·m². A non-diagonal inertia tensor has to be
/// diagonalised by the caller; this will not pretend a tensor's diagonal is its spectrum.
#[pyfunction]
fn worst_principal_axis(
    law: &str,
    proportional: f64,
    derivative: f64,
    moments: [f64; 3],
) -> PyResult<(usize, f64, f64)> {
    let law = AttitudeLawKind::build(law, proportional, derivative)?;
    let (axis, margin) = control_worst_axis(&law, moments).ok_or_else(|| {
        PyValueError::new_err("every principal moment must be finite and strictly positive")
    })?;
    Ok((axis, margin.seconds(), margin.crossing_frequency()))
}

/// The Clohessy–Wiltshire relative plant at `mean_motion` rad/s, as `(a, b)` flattened row-major:
/// `a` is 6×6 and `b` is 6×3, in the LVLH state order `[x, y, z, ẋ, ẏ, ż]`.
#[pyfunction]
fn clohessy_wiltshire_plant(mean_motion: f64) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (a, b) = control_cw_plant(mean_motion)
        .ok_or_else(|| PyValueError::new_err("the mean motion must be finite and non-zero"))?;
    Ok((flatten(&a), flatten(&b)))
}

/// The infinite-horizon LQR gain for `ẋ = Ax + Bu` under `J = ∫ (xᵀQx + uᵀRu) dt`.
///
/// All four arguments are flat row-major: `a` and `q` are `N×N`, `r` is `M×M`, and `b` is `N×M`
/// indexed `b[state][control]`. Returns `(gain, riccati, residual)` — `K` flattened `M×N` so that
/// `u = −Kx`, `P` flattened `N×N`, and the relative Riccati residual the solution was accepted at,
/// which is the measured accuracy of the gain rather than a promise about it.
///
/// The Rust `lqr` is const-generic in the state and control dimensions, so a Python caller reaches
/// only the shapes this binding was monomorphised for: `(N, M)` in `(2,1)`, `(3,1)`, `(4,2)`,
/// `(6,3)`. The last is the Clohessy–Wiltshire plant and the first the double integrator.
#[pyfunction]
fn lqr(a: Vec<f64>, b: Vec<f64>, q: Vec<f64>, r: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>, f64)> {
    let n = a.len().isqrt();
    let m = r.len().isqrt();
    if n * n != a.len() || m * m != r.len() {
        return Err(PyValueError::new_err(
            "`a` and `r` must each be a square matrix flattened row-major",
        ));
    }
    match (n, m) {
        (2, 1) => solve_lqr::<2, 1, 4>(&a, &b, &q, &r),
        (3, 1) => solve_lqr::<3, 1, 6>(&a, &b, &q, &r),
        (4, 2) => solve_lqr::<4, 2, 8>(&a, &b, &q, &r),
        (6, 3) => solve_lqr::<6, 3, 12>(&a, &b, &q, &r),
        _ => Err(PyValueError::new_err(format!(
            "the LQR binding is monomorphised for (n, m) in (2,1), (3,1), (4,2), (6,3), not ({n}, {m})"
        ))),
    }
}

fn solve_lqr<const N: usize, const M: usize, const N2: usize>(
    a: &[f64],
    b: &[f64],
    q: &[f64],
    r: &[f64],
) -> PyResult<(Vec<f64>, Vec<f64>, f64)> {
    let solved = control_lqr::<N, M, N2>(
        &rows_of::<N, N>(a)?,
        &rows_of::<M, N>(b)?,
        &rows_of::<N, N>(q)?,
        &rows_of::<M, M>(r)?,
    )
    .map_err(|e| PyValueError::new_err(format!("no optimal gain: {e:?}")))?;
    Ok((
        flatten(solved.gain()),
        flatten(solved.riccati()),
        solved.residual(),
    ))
}

/// `V = ½ Δœᵀ W Δœ` for the station-keeping law, with `weights` the six Bryson weights.
#[pyfunction]
fn element_feedback_lyapunov(
    weights: [f64; 6],
    proportional: f64,
    max_acceleration: f64,
    error: [f64; 6],
) -> PyResult<f64> {
    Ok(element_feedback(weights, proportional, max_acceleration)?.lyapunov(&error))
}

fn element_feedback(
    weights: [f64; 6],
    proportional: f64,
    max_acceleration: f64,
) -> PyResult<ElementFeedback> {
    ElementFeedback::new(weights, proportional, max_acceleration).ok_or_else(|| {
        PyValueError::new_err(
            "every weight must be finite and non-negative with one positive, and the gain and the \
             acceleration limit finite and strictly positive",
        )
    })
}

/// The station-keeping acceleration in the inertial orbit frame, m/s² — the frame every DSST force
/// returns its acceleration in, so this sums with them directly.
///
/// `elements` are the osculating equinoctial elements the variational matrix is evaluated at, in
/// the DSST's `[a, k, h, q, p, λ]` order with `a` in metres, and `error` is `œ − œ_target` in the
/// same order. The mean-longitude component of the error is wrapped to `(−π, π]` before use: an
/// unwrapped `λ` error of `2π − ε` would command nearly a full revolution of drift to reach an orbit
/// that is `ε` away.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn element_feedback_acceleration(
    weights: [f64; 6],
    proportional: f64,
    max_acceleration: f64,
    elements: [f64; 6],
    mu: f64,
    true_longitude: f64,
    error: [f64; 6],
) -> PyResult<[f64; 3]> {
    let law = element_feedback(weights, proportional, max_acceleration)?;
    let aux = auxiliary_of(elements, mu);
    Ok(law.acceleration(&aux, true_longitude, &error))
}

/// `V̇` along the closed loop, which the station-keeping law makes non-positive by construction.
///
/// Computed from the same `B(œ)` the command used, so a caller can measure the decrease rather than
/// take the derivation's word for it. Arguments as [`element_feedback_acceleration`].
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn element_feedback_decrease_rate(
    weights: [f64; 6],
    proportional: f64,
    max_acceleration: f64,
    elements: [f64; 6],
    mu: f64,
    true_longitude: f64,
    error: [f64; 6],
) -> PyResult<f64> {
    let law = element_feedback(weights, proportional, max_acceleration)?;
    let aux = auxiliary_of(elements, mu);
    Ok(law.decrease_rate(&aux, true_longitude, &error))
}

fn auxiliary_of(elements: [f64; 6], mu: f64) -> DsstAuxiliary {
    DsstAuxiliary::new(
        elements[0],
        elements[1],
        elements[2],
        elements[3],
        elements[4],
        elements[5],
        mu,
    )
}

/// Run `call` at the square dimension `$agents`, which is `$what` in the refusal.
///
/// `Graph<N>` and `characteristic_polynomial` are both const-generic in that dimension, so the
/// bindings carry one instantiation per size. The range is 2 to 12: two is the smallest graph that
/// can disagree, and each instantiation past twelve costs a monomorphised eigensolver and delayed
/// integrator for a formation size nobody has asked for. `$what` is a parameter because the two
/// walls are the same number and not the same wall — a 13-agent graph and a 13x13 matrix are
/// refused for different reasons and a caller reading the message should be told which they hit.
macro_rules! dispatch_agents {
    ($what:literal, $agents:expr, $call:ident $(, $arg:expr)*) => {
        match $agents {
            2 => $call::<2>($($arg),*),
            3 => $call::<3>($($arg),*),
            4 => $call::<4>($($arg),*),
            5 => $call::<5>($($arg),*),
            6 => $call::<6>($($arg),*),
            7 => $call::<7>($($arg),*),
            8 => $call::<8>($($arg),*),
            9 => $call::<9>($($arg),*),
            10 => $call::<10>($($arg),*),
            11 => $call::<11>($($arg),*),
            12 => $call::<12>($($arg),*),
            n => Err(PyValueError::new_err(format!(
                "the control bindings are monomorphised for 2 to 12 {}, not {n}",
                $what
            ))),
        }
    };
}

fn agents_of(adjacency: &[f64]) -> PyResult<usize> {
    let n = adjacency.len().isqrt();
    if n * n != adjacency.len() {
        return Err(PyValueError::new_err(format!(
            "the adjacency matrix must be square and flattened row-major, and {} is not a square",
            adjacency.len()
        )));
    }
    Ok(n)
}

fn graph_of<const N: usize>(adjacency: &[f64]) -> PyResult<Graph<N>> {
    Graph::new(rows_of::<N, N>(adjacency)?)
        .map_err(|e| PyValueError::new_err(format!("not a usable interaction graph: {e:?}")))
}

/// The Laplacian spectrum of an interaction graph, ascending.
///
/// `adjacency` is the flat row-major weight matrix; the weights are control gains, so they need not
/// be one. `ValueError` unless the graph is symmetric, non-negative, hollow and connected — a
/// disconnected graph has a repeated zero eigenvalue and never reaches consensus at all, which is a
/// different answer from a small one.
#[pyfunction]
fn laplacian_spectrum(adjacency: Vec<f64>) -> PyResult<Vec<f64>> {
    fn call<const N: usize>(adjacency: &[f64]) -> PyResult<Vec<f64>> {
        Ok(graph_of::<N>(adjacency)?.laplacian_spectrum().to_vec())
    }
    dispatch_agents!("agents", agents_of(&adjacency)?, call, &adjacency)
}

/// The Laplacian eigenvalues ascending, and their unit eigenvectors flattened row-major as
/// `modes[k]`.
///
/// The eigenvectors are what [`laplacian_spectrum`] leaves behind, and they are what the
/// disagreement decomposes into: each mode evolves as the scalar delay equation `ẏ = −λₖ y(t−τ)`
/// under `'delayed_relative'`, independently of every other. Watching the mode that crosses first
/// is not the same as watching whichever mode happens to dominate the aggregate norm.
#[pyfunction]
fn laplacian_modes(adjacency: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    fn call<const N: usize>(adjacency: &[f64]) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let (values, modes) = graph_of::<N>(adjacency)?.laplacian_eigen();
        Ok((values.to_vec(), flatten(&modes)))
    }
    dispatch_agents!("agents", agents_of(&adjacency)?, call, &adjacency)
}

/// Whether every agent can reach every other over links of non-zero gain.
///
/// Combinatorial, by breadth-first search, and therefore exact: connectivity is a property of the
/// sparsity pattern, where `λ₂ > tolerance` on [`algebraic_connectivity`] would need a tolerance
/// nobody can choose for an arbitrary weighting.
#[pyfunction]
fn graph_is_connected(adjacency: Vec<f64>) -> PyResult<bool> {
    // `Graph::new` refuses a disconnected graph, and `is_connected` is only reachable through it —
    // so the answer this binding exists to give arrives as that refusal, and reading it is the
    // whole of the function. Every other `GraphError` stays an exception: a matrix that is not
    // symmetric is not a disconnected graph, it is not a graph.
    fn call<const N: usize>(adjacency: &[f64]) -> PyResult<bool> {
        match Graph::<N>::new(rows_of::<N, N>(adjacency)?) {
            Ok(_) => Ok(true),
            Err(GraphError::Disconnected) => Ok(false),
            Err(e) => Err(PyValueError::new_err(format!(
                "not a usable interaction graph: {e:?}"
            ))),
        }
    }
    dispatch_agents!("agents", agents_of(&adjacency)?, call, &adjacency)
}

/// The algebraic connectivity `λ₂` — the convergence rate of undelayed consensus on this graph.
#[pyfunction]
fn algebraic_connectivity(adjacency: Vec<f64>) -> PyResult<f64> {
    fn call<const N: usize>(adjacency: &[f64]) -> PyResult<f64> {
        Ok(graph_of::<N>(adjacency)?.algebraic_connectivity())
    }
    dispatch_agents!("agents", agents_of(&adjacency)?, call, &adjacency)
}

/// The monic characteristic polynomial of a square matrix: the coefficients of `λ⁰ … λ^{N−1}`.
///
/// The `λ^N` coefficient is 1 and is not returned. `m` is flattened row-major and its order is the
/// square root of its length.
///
/// This is what makes a gain from [`lqr`] checkable without an eigensolver: the closed-loop matrix
/// `A − BK` is stable exactly when this polynomial's roots are all in the left half-plane, and the
/// coefficients are real and order-free where the eigenvalues are neither — which is what lets them
/// be compared against a CAS directly. Faddeev–LeVerrier, so the cost is matrix products and traces.
///
/// The recurrence divides by `k` and so loses roughly `N` digits of the smallest coefficient on a
/// matrix whose eigenvalues are widely spread; on the 6×6 closed loops here that is measured at
/// `1e-12` relative against a 40-digit reference, and for a much larger or stiffer matrix it would
/// be the wrong tool.
#[pyfunction]
fn characteristic_polynomial(m: Vec<f64>) -> PyResult<Vec<f64>> {
    fn call<const N: usize>(m: &[f64]) -> PyResult<Vec<f64>> {
        Ok(control_characteristic_polynomial(&rows_of::<N, N>(m)?).to_vec())
    }
    let order = m.len().isqrt();
    if order * order != m.len() {
        return Err(PyValueError::new_err(format!(
            "the matrix must be square and flattened row-major, and {} is not a square",
            m.len()
        )));
    }
    dispatch_agents!("rows", order, call, &m)
}

fn protocol_of(protocol: &str) -> PyResult<Protocol> {
    match protocol {
        "delayed_relative" => Ok(Protocol::DelayedRelative),
        "current_self" => Ok(Protocol::CurrentSelf),
        other => Err(PyValueError::new_err(format!(
            "protocol must be 'delayed_relative' or 'current_self', not {other:?}"
        ))),
    }
}

/// The largest link delay at which a protocol still reaches consensus, seconds.
///
/// `'delayed_relative'` — the relative measurement itself is late, so both terms are equally old —
/// is the protocol Olfati-Saber & Murray bound at `π / (2 λ_max(L))`, and that is what comes back.
/// `'current_self'`, where the agent's own state is current and only the neighbour's is late,
/// returns `inf`: no delay destroys its stability. The convergence *rate* still falls with delay in
/// both; what is unbounded is the stability, not the performance.
#[pyfunction]
fn consensus_delay_bound(adjacency: Vec<f64>, protocol: &str) -> PyResult<f64> {
    fn call<const N: usize>(adjacency: &[f64], protocol: Protocol) -> PyResult<f64> {
        let graph = graph_of::<N>(adjacency)?;
        // The bound does not read the delay, so any admissible one produces the same answer.
        let net = ConsensusNetwork::new(graph, protocol, 1.0)
            .ok_or_else(|| PyValueError::new_err("unreachable: 1.0 s is an admissible delay"))?;
        Ok(net.delay_bound().seconds())
    }
    let protocol = protocol_of(protocol)?;
    dispatch_agents!("agents", agents_of(&adjacency)?, call, &adjacency, protocol)
}

/// The disagreement `‖x − x̄1‖`, which is what consensus drives to zero.
///
/// The mean is subtracted rather than a designated leader's value, because no agent is a leader and
/// the mean is the quantity the dynamics conserve.
#[pyfunction]
fn consensus_disagreement(state: Vec<f64>) -> PyResult<f64> {
    fn call<const N: usize>(state: &[f64]) -> PyResult<f64> {
        Ok(ConsensusNetwork::<N>::disagreement(&vector_of::<N>(state)?))
    }
    dispatch_agents!("agents", state.len(), call, &state)
}

/// Integrate delayed consensus over a fixed graph, by the method of steps.
///
/// Returns `(final_state, disagreement, accepted, rejected)`: the agents' states at `t1`, the
/// disagreement there, and the accepted and rejected step counts summed over every arc of the mesh.
/// A breakpoint is placed at every propagated discontinuity `kτ` — a delayed system's solution has a
/// jump in `x'` at the origin, in `x''` at `τ` and in `x'''` at `2τ`, and a mesh that steps over one
/// loses the integrator's order.
///
/// `delay` must be finite and strictly positive: a zero delay is an ordinary differential equation
/// and belongs to a different driver, and the method of steps needs a positive shortest delay to
/// bound its mesh.
#[pyfunction]
fn consensus_run(
    py: Python<'_>,
    adjacency: Vec<f64>,
    protocol: &str,
    delay: f64,
    initial: Vec<f64>,
    t1: f64,
    tolerance: f64,
) -> PyResult<(Vec<f64>, f64, u32, u32)> {
    fn call<const N: usize>(
        py: Python<'_>,
        adjacency: &[f64],
        protocol: Protocol,
        delay: f64,
        initial: &[f64],
        t1: f64,
        tolerance: f64,
    ) -> PyResult<(Vec<f64>, f64, u32, u32)> {
        let graph = graph_of::<N>(adjacency)?;
        let net = ConsensusNetwork::new(graph, protocol, delay).ok_or_else(|| {
            PyValueError::new_err("the link delay must be finite and strictly positive")
        })?;
        let y0 = vector_of::<N>(initial)?;
        let solution = py
            .detach(|| {
                integrate_delayed(
                    &Dop853,
                    &net,
                    InitialValue::at_rest(0.0, y0),
                    t1,
                    0.05 * delay,
                    Controls::new(Tolerance::new(tolerance, tolerance), 1e-10),
                )
            })
            .map_err(|e| PyValueError::new_err(format!("the delayed run failed: {e:?}")))?;
        let final_state = solution.final_state();
        let (accepted, rejected) = solution.step_counts();
        Ok((
            final_state.to_vec(),
            ConsensusNetwork::<N>::disagreement(&final_state),
            accepted,
            rejected,
        ))
    }
    let protocol = protocol_of(protocol)?;
    let agents = agents_of(&adjacency)?;
    if initial.len() != agents {
        return Err(PyValueError::new_err(format!(
            "the graph has {agents} agents and the initial state {}",
            initial.len()
        )));
    }
    dispatch_agents!(
        "agents", agents, call, py, &adjacency, protocol, delay, &initial, t1, tolerance
    )
}

/// Every propagated discontinuity of a delayed system strictly inside `(t0, t1)`, sorted and merged.
///
/// A delayed system smooths one order per delay interval, so the jumps sit at every sum of `levels`
/// or fewer delays. These are the times an adaptive mesh has to include; the merge tolerance is
/// relative to the span, and a breakpoint floating-point absorbed into an endpoint is dropped rather
/// than emitted as a zero-length arc.
#[pyfunction]
fn delay_breakpoints(delays: Vec<f64>, t0: f64, t1: f64, levels: usize) -> PyResult<Vec<f64>> {
    breakpoints(&delays, t0, t1, levels).ok_or_else(|| {
        PyValueError::new_err(
            "every delay must be finite and strictly positive, `t1` must follow `t0`, and the \
             breakpoint count must stay inside its cap",
        )
    })
}

/// Adds every binding to `m`, the Python module of the cdylib that links this crate.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Sensor>()?;
    m.add_class::<PyDensityRom>()?;
    m.add_class::<PyDmdc>()?;
    m.add_class::<PyTimeVariableGravity>()?;
    m.add_class::<PyReducedDynamics>()?;
    m.add_class::<Srif6>()?;
    m.add_class::<DenseEphemeris>()?;
    m.add_class::<Ias15Ephemeris>()?;
    m.add_class::<De440>()?;
    m.add_function(wrap_pyfunction!(walker_constellation, m)?)?;
    m.add_function(wrap_pyfunction!(street_of_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(street_of_coverage_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(flower_constellation, m)?)?;
    m.add_function(wrap_pyfunction!(lattice_flower, m)?)?;
    m.add_function(wrap_pyfunction!(draim_constellation, m)?)?;
    m.add_function(wrap_pyfunction!(draim_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_characteristic, m)?)?;
    m.add_function(wrap_pyfunction!(minimum_walker, m)?)?;
    m.add_function(wrap_pyfunction!(gauss_newton, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_geopotential_dense, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_geopotential_itrf_dense, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_j2_itrf_dense, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_j2_itrf_eop_dense, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_geopotential_tide_dense, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_relativistic_dense, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_drag_dense, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_drag_nrlmsise00_dense, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_drag_sdtm1_dense, m)?)?;
    m.add_function(wrap_pyfunction!(nrlmsise00_density, m)?)?;
    m.add_function(wrap_pyfunction!(sdtm1_density, m)?)?;
    m.add_function(wrap_pyfunction!(sdtm1_exospheric_temperature, m)?)?;
    m.add_function(wrap_pyfunction!(sdtm1_storm_response, m)?)?;
    m.add_function(wrap_pyfunction!(sdtm1_density_interval, m)?)?;
    m.add_function(wrap_pyfunction!(sdtm1_envelope, m)?)?;
    m.add_function(wrap_pyfunction!(brouwer_lyddane_propagate, m)?)?;
    m.add_function(wrap_pyfunction!(dsst_zonal_propagate_mean, m)?)?;
    m.add_function(wrap_pyfunction!(dsst_zonal_propagate_osculating, m)?)?;
    m.add_function(wrap_pyfunction!(dsst_propagate_mean_full, m)?)?;
    m.add_function(wrap_pyfunction!(dsst_propagate_osculating_full, m)?)?;
    m.add_function(wrap_pyfunction!(solid_tide_step2, m)?)?;
    m.add_function(wrap_pyfunction!(solid_pole_tide, m)?)?;
    m.add_function(wrap_pyfunction!(ocean_pole_tide, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_luni_solar_dense, m)?)?;
    m.add_function(wrap_pyfunction!(sgp4_propagate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_dense, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_ias15_dense, m)?)?;
    m.add_function(wrap_pyfunction!(range, m)?)?;
    m.add_function(wrap_pyfunction!(range_rate, m)?)?;
    m.add_function(wrap_pyfunction!(bistatic_range, m)?)?;
    m.add_function(wrap_pyfunction!(bistatic_range_rate, m)?)?;
    m.add_function(wrap_pyfunction!(tdoa, m)?)?;
    m.add_function(wrap_pyfunction!(fdoa, m)?)?;
    m.add_function(wrap_pyfunction!(inter_satellite_range, m)?)?;
    m.add_function(wrap_pyfunction!(inter_satellite_range_rate, m)?)?;
    m.add_function(wrap_pyfunction!(dsn_sequential_range, m)?)?;
    m.add_function(wrap_pyfunction!(dsn_averaged_doppler, m)?)?;
    m.add_function(wrap_pyfunction!(dsn_measured_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(delta_dor, m)?)?;
    m.add_function(wrap_pyfunction!(solar_corona_delay, m)?)?;
    m.add_function(wrap_pyfunction!(orbital_lifetime, m)?)?;
    m.add_function(wrap_pyfunction!(orbital_lifetime_ensemble, m)?)?;
    m.add_function(wrap_pyfunction!(ballistic_log_sensitivity, m)?)?;
    m.add_function(wrap_pyfunction!(tle_lines, m)?)?;
    m.add_function(wrap_pyfunction!(rinex_obs, m)?)?;
    m.add_function(wrap_pyfunction!(rinex_obs_series, m)?)?;
    m.add_function(wrap_pyfunction!(carrier_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(lane_wavelengths, m)?)?;
    m.add_function(wrap_pyfunction!(ionosphere_free, m)?)?;
    m.add_function(wrap_pyfunction!(wide_lane, m)?)?;
    m.add_function(wrap_pyfunction!(narrow_lane, m)?)?;
    m.add_function(wrap_pyfunction!(melbourne_wubbena, m)?)?;
    m.add_function(wrap_pyfunction!(geometry_free, m)?)?;
    m.add_function(wrap_pyfunction!(graphic, m)?)?;
    m.add_function(wrap_pyfunction!(detect_cycle_slips_geometry_free, m)?)?;
    m.add_function(wrap_pyfunction!(detect_cycle_slips_polynomial, m)?)?;
    m.add_function(wrap_pyfunction!(ambiguity_decorrelation, m)?)?;
    m.add_function(wrap_pyfunction!(ambiguity_bootstrap, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_ambiguities, m)?)?;
    m.add_function(wrap_pyfunction!(collision_probability, m)?)?;
    m.add_function(wrap_pyfunction!(conjunction_usage_violations, m)?)?;
    m.add_function(wrap_pyfunction!(closest_approaches, m)?)?;
    m.add_function(wrap_pyfunction!(radially_separated, m)?)?;
    m.add_function(wrap_pyfunction!(shapiro_delay, m)?)?;
    m.add_function(wrap_pyfunction!(elevation, m)?)?;
    m.add_function(wrap_pyfunction!(azimuth, m)?)?;
    m.add_function(wrap_pyfunction!(grid_cells, m)?)?;
    m.add_function(wrap_pyfunction!(coverage_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(constellation_fom, m)?)?;
    m.add_function(wrap_pyfunction!(access_windows, m)?)?;
    m.add_function(wrap_pyfunction!(teme_to_itrf, m)?)?;
    m.add_function(wrap_pyfunction!(gcrf_to_itrf, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_two_body, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_j2_precision, m)?)?;
    m.add_function(wrap_pyfunction!(itrf_to_geodetic, m)?)?;
    m.add_function(wrap_pyfunction!(geodetic_to_itrf, m)?)?;
    m.add_function(wrap_pyfunction!(sgp4_propagate, m)?)?;
    m.add_function(wrap_pyfunction!(kepler_propagate, m)?)?;
    m.add_function(wrap_pyfunction!(lambert, m)?)?;
    m.add_function(wrap_pyfunction!(richardson_halo, m)?)?;
    m.add_function(wrap_pyfunction!(geopotential_acceleration, m)?)?;
    m.add_function(wrap_pyfunction!(libration_points, m)?)?;
    m.add_function(wrap_pyfunction!(rinex_gnss_positions, m)?)?;
    m.add_function(wrap_pyfunction!(rinex_glonass_positions, m)?)?;
    m.add_function(wrap_pyfunction!(dilution_of_precision, m)?)?;
    m.add_function(wrap_pyfunction!(klobuchar_iono_delay, m)?)?;
    m.add_function(wrap_pyfunction!(saastamoinen_zenith_delay, m)?)?;
    m.add_function(wrap_pyfunction!(sun_synchronous_inclination, m)?)?;
    m.add_function(wrap_pyfunction!(repeat_ground_track_sma, m)?)?;
    m.add_function(wrap_pyfunction!(frozen_eccentricity, m)?)?;
    m.add_function(wrap_pyfunction!(critical_inclination, m)?)?;
    m.add_function(wrap_pyfunction!(nequick_galileo_stec, m)?)?;
    m.add_function(wrap_pyfunction!(refraction_apparent, m)?)?;
    m.add_function(wrap_pyfunction!(refraction_geometric, m)?)?;
    m.add_function(wrap_pyfunction!(diurnal_velocity, m)?)?;
    m.add_function(wrap_pyfunction!(aberration_apply, m)?)?;
    m.add_function(wrap_pyfunction!(aberration_remove, m)?)?;
    m.add_function(wrap_pyfunction!(ionospheric_path_delay, m)?)?;
    m.add_function(wrap_pyfunction!(second_order_ionospheric_delay, m)?)?;
    m.add_function(wrap_pyfunction!(igrf_field_ecef, m)?)?;
    m.add_function(wrap_pyfunction!(niell_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(gmf_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(vmf1_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(vmf3_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(vmf_grid_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(gpt3, m)?)?;
    m.add_function(wrap_pyfunction!(density_correction_fit, m)?)?;
    m.add_function(wrap_pyfunction!(density_correction_factor, m)?)?;
    m.add_function(wrap_pyfunction!(density_data_samples, m)?)?;
    m.add_function(wrap_pyfunction!(density_altitude_normalized, m)?)?;
    m.add_function(wrap_pyfunction!(density_grid_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(space_weather_days, m)?)?;
    m.add_function(wrap_pyfunction!(space_weather_drivers, m)?)?;
    m.add_function(wrap_pyfunction!(space_weather_ap_array, m)?)?;
    m.add_function(wrap_pyfunction!(sentman_drag_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(molecular_speed_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(time_variable_zonal_correction, m)?)?;
    m.add_function(wrap_pyfunction!(conventional_zonal_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(glonass_l1_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(glonass_l2_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(swath_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(horizon_look_angle, m)?)?;
    m.add_function(wrap_pyfunction!(ground_point_at, m)?)?;
    m.add_class::<PyRpc>()?;
    m.add_function(wrap_pyfunction!(atmospheric_surface_from_toa, m)?)?;
    m.add_function(wrap_pyfunction!(atmospheric_toa_from_surface, m)?)?;
    m.add_function(wrap_pyfunction!(brdf_reflectance, m)?)?;
    m.add_function(wrap_pyfunction!(ross_thick_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(li_sparse_reciprocal_kernel, m)?)?;
    // Phase-8 spacesignal.
    m.add_function(wrap_pyfunction!(wrap_phase, m)?)?;
    m.add_function(wrap_pyfunction!(residues, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_phase_1d, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_phase_2d, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_phase_2d_lstsq, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_phase_2d_branch_cut, m)?)?;
    m.add_function(wrap_pyfunction!(interferogram, m)?)?;
    m.add_function(wrap_pyfunction!(coherence, m)?)?;
    m.add_function(wrap_pyfunction!(lfm_chirp, m)?)?;
    m.add_function(wrap_pyfunction!(matched_filter, m)?)?;
    m.add_function(wrap_pyfunction!(resolution_3db_samples, m)?)?;
    m.add_function(wrap_pyfunction!(peak_sidelobe_ratio_db, m)?)?;
    m.add_function(wrap_pyfunction!(azimuth_fm_rate, m)?)?;
    m.add_function(wrap_pyfunction!(range_migration, m)?)?;
    m.add_function(wrap_pyfunction!(src_fm_rate, m)?)?;
    m.add_function(wrap_pyfunction!(bartlett_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(capon_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(sl1mmer_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(sl1mmer, m)?)?;
    // Phase-8 earthvision v2–v4.
    m.add_function(wrap_pyfunction!(semi_global_match, m)?)?;
    m.add_function(wrap_pyfunction!(harris_corners, m)?)?;
    m.add_function(wrap_pyfunction!(rpc_block_adjust, m)?)?;
    // Phase-8 deepvision detection.
    m.add_function(wrap_pyfunction!(nms, m)?)?;
    m.add_function(wrap_pyfunction!(box_iou, m)?)?;
    m.add_function(wrap_pyfunction!(generate_anchors, m)?)?;
    m.add_function(wrap_pyfunction!(decode_box, m)?)?;
    // Phase-8 v5.
    m.add_class::<PyCamera>()?;
    m.add_function(wrap_pyfunction!(triangulate, m)?)?;
    m.add_function(wrap_pyfunction!(plane_sweep_mvs, m)?)?;
    m.add_function(wrap_pyfunction!(sfm_triangulate, m)?)?;
    m.add_function(wrap_pyfunction!(brovey_pan_sharpen, m)?)?;
    m.add_function(wrap_pyfunction!(gram_schmidt_pan_sharpen, m)?)?;
    m.add_function(wrap_pyfunction!(pauli_powers, m)?)?;
    m.add_function(wrap_pyfunction!(coherency_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(cloude_pottier, m)?)?;
    m.add_function(wrap_pyfunction!(amplitude_dispersion, m)?)?;
    m.add_function(wrap_pyfunction!(select_persistent_scatterers, m)?)?;
    m.add_function(wrap_pyfunction!(sbas_invert, m)?)?;
    m.add_function(wrap_pyfunction!(apply_chirp_scaling, m)?)?;
    m.add_function(wrap_pyfunction!(chirp_scaling_focus, m)?)?;
    #[cfg(feature = "onnx")]
    m.add_function(wrap_pyfunction!(cloud_mask, m)?)?;
    // The control pillar.
    m.add_function(wrap_pyfunction!(latency_total, m)?)?;
    m.add_function(wrap_pyfunction!(latency_steps, m)?)?;
    m.add_function(wrap_pyfunction!(attitude_error_mrp, m)?)?;
    m.add_function(wrap_pyfunction!(attitude_error_quaternion, m)?)?;
    m.add_function(wrap_pyfunction!(attitude_torque, m)?)?;
    m.add_function(wrap_pyfunction!(attitude_lyapunov, m)?)?;
    m.add_function(wrap_pyfunction!(attitude_small_angle_gains, m)?)?;
    m.add_function(wrap_pyfunction!(quaternion_pd_critically_damped, m)?)?;
    m.add_function(wrap_pyfunction!(quaternion_pd_response, m)?)?;
    m.add_function(wrap_pyfunction!(attitude_loop_run, m)?)?;
    m.add_function(wrap_pyfunction!(delay_margin_of_gains, m)?)?;
    m.add_function(wrap_pyfunction!(delay_margin_of_law, m)?)?;
    m.add_function(wrap_pyfunction!(worst_principal_axis, m)?)?;
    m.add_function(wrap_pyfunction!(clohessy_wiltshire_plant, m)?)?;
    m.add_function(wrap_pyfunction!(lqr, m)?)?;
    m.add_function(wrap_pyfunction!(characteristic_polynomial, m)?)?;
    m.add_function(wrap_pyfunction!(element_feedback_lyapunov, m)?)?;
    m.add_function(wrap_pyfunction!(element_feedback_acceleration, m)?)?;
    m.add_function(wrap_pyfunction!(element_feedback_decrease_rate, m)?)?;
    m.add_function(wrap_pyfunction!(laplacian_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(laplacian_modes, m)?)?;
    m.add_function(wrap_pyfunction!(graph_is_connected, m)?)?;
    m.add_function(wrap_pyfunction!(algebraic_connectivity, m)?)?;
    m.add_function(wrap_pyfunction!(consensus_delay_bound, m)?)?;
    m.add_function(wrap_pyfunction!(consensus_disagreement, m)?)?;
    m.add_function(wrap_pyfunction!(consensus_run, m)?)?;
    m.add_function(wrap_pyfunction!(delay_breakpoints, m)?)?;
    m.add(
        "__doc__",
        "Python bindings for the spacelab space-data computation core.",
    )?;
    Ok(())
}

/// The bridge's own logic, as opposed to the crates it marshals into.
///
/// `cargo mutants` drives `cargo test`, and until this module existed the crate had none, so every
/// one of its mutants was recorded MISSED for one reason. The first tranche is deliberately scoped
/// to what a mutation of *this file* can break and the core crates cannot catch: the `Vec3`
/// packing, `lerp_at`'s clamped interpolation, the degree/radian conversions, the grid's iteration
/// order, and the frame conventions of the measurement models.
///
/// The second tranche is the five functions whose *return type* dominates the plan. `--list` offers
/// 19 239 mutants for this one file and 16 529 of them — 86% — sit in five functions, because a
/// whole-function replacement over a wide tuple is enumerated as the Cartesian product of its
/// components: `street_of_coverage_geometry` returns `Option<(u32, u32, f64 × 7)>` and so is
/// listed `1 + 2·2·3⁷ = 8749` times, `constellation_fom` `2·3⁶·2 = 2916`, `space_weather_days` and
/// `density_altitude_normalized` `4·3²·4³ = 2304` each, `density_data_samples` `4⁴ = 256`.
///
/// Those are not 16 529 questions. Each function is asked one question — *does any test pin what
/// this returns?* — once per point of a grid, and one `assert_eq!` on the whole tuple answers every
/// point of that grid the same way. The five tests below are those assertions; the ledger in
/// `.cargo/mutants.toml` then collapses each product to a single surviving representative, so
/// deleting a test still turns the gate red while the block costs 2 715 mutants rather than 19 239.
///
/// It stops where the link does. `pyo3` is depended on with `extension-module`, which is correct
/// for a `cdylib` — the interpreter supplies `Py*` at import time — and which leaves a *test*
/// binary with undefined symbols the moment anything reaches a `PyErr`. So every `PyResult`-
/// returning entry point (`itrf_to_geodetic`, `geopotential_acceleration`, `richardson_halo`,
/// `sgp4_propagate`, every `propagate_*_dense`) is still out of reach here, and putting it in reach
/// means making `extension-module` an optional default feature and running these under
/// `--no-default-features` — a change to how the shipped wheel is configured, not a test.
#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2};

    /// WGS-84 semi-minor axis, metres — `a(1 − f)` with `f = 1/298.257223563`, to the precision the
    /// defining constants give.
    const WGS84_B: f64 = 6_356_752.314_245_179;

    #[test]
    fn the_vector_conversion_keeps_its_component_order() {
        let v = vec3([1.0, 2.0, 3.0]);
        assert_eq!((v.x, v.y, v.z), (1.0, 2.0, 3.0));
        assert_eq!(arr(v), [1.0, 2.0, 3.0]);
        // Distinct, non-symmetric components on purpose: a swapped pair or a dropped one shows.
        assert_eq!(arr(vec3([-7.5, 0.25, 1e9])), [-7.5, 0.25, 1e9]);
    }

    /// The clamped linear interpolation an EOP series is fed through. Three separate behaviours —
    /// the two clamps, the nodes, and the interior — and the bracket index is `partition_point − 1`,
    /// so a series with unequal spacing is what tells a correct bracket from an off-by-one.
    ///
    /// One mutation of this function no test can kill, and it is equivalent rather than uncovered:
    /// the *lower* clamp `t <= ts[0]` tied to `t < ts[0]`. On a strictly increasing series the
    /// interior branch reproduces the first node anyway — `partition_point(|&x| x <= ts[0]) − 1` is
    /// `0` and `frac` is exactly `0`, giving `ys[0]` — so the two forms differ only on a series
    /// whose first node is repeated, which is not a series. The upper clamp is *not* redundant in
    /// the same way, because falling through it indexes `ts[n]`; that one is killed above.
    #[test]
    fn the_series_interpolation_clamps_at_both_ends_and_is_exact_on_its_nodes() {
        let ts = [10.0, 20.0, 40.0];
        let ys = [1.0, 3.0, 4.0];

        assert_eq!(lerp_at(&ts, &ys, -1e9), 1.0, "below the first node");
        assert_eq!(lerp_at(&ts, &ys, 9.999), 1.0, "just below the first node");
        assert_eq!(lerp_at(&ts, &ys, 1e9), 4.0, "above the last node");
        assert_eq!(lerp_at(&ts, &ys, 40.001), 4.0, "just above the last node");

        for (t, y) in ts.iter().zip(ys.iter()) {
            assert_eq!(lerp_at(&ts, &ys, *t), *y, "node {t} is not exact");
        }

        // Unequal spacing: the two intervals have slopes 0.2 and 0.05, so a bracket off by one
        // cannot reproduce both midpoints.
        assert!((lerp_at(&ts, &ys, 15.0) - 2.0).abs() < 1e-15);
        assert!((lerp_at(&ts, &ys, 30.0) - 3.5).abs() < 1e-15);
        assert!((lerp_at(&ts, &ys, 12.5) - 1.5).abs() < 1e-15);

        // A one-sample series is both clamps at once — the degenerate case an EOP file of one day is.
        assert_eq!(lerp_at(&[7.0], &[9.0], -3.0), 9.0);
        assert_eq!(lerp_at(&[7.0], &[9.0], 7.0), 9.0);
        assert_eq!(lerp_at(&[7.0], &[9.0], 11.0), 9.0);
    }

    /// The ground grid is degrees in, ECEF out, latitude outer and longitude inner — and the centre
    /// is the zenith scaled by one Earth radius, which is what `coverage_metrics` relies on to treat
    /// the pair as a cell rather than as two independent vectors.
    #[test]
    fn the_ground_grid_is_row_major_in_degrees_on_a_sphere() {
        let cells = grid_cells(vec![0.0, 45.0], vec![0.0, 90.0]);
        assert_eq!(cells.len(), 4, "the grid is the product of its two axes");

        // Row-major: (0,0), (0,90), (45,0), (45,90). Reading it column-major would put (45,0) second.
        assert_eq!(cells[0].0, [RE_EARTH, 0.0, 0.0]);
        assert!(
            cells[1].0[1] > 6.3e6,
            "cell 1 is not on the 90° meridian: {:?}",
            cells[1].0
        );
        assert!(
            cells[2].0[2] > 4.5e6,
            "cell 2 is not at 45° north: {:?}",
            cells[2].0
        );

        for (centre, zenith) in &cells {
            let n = (zenith[0] * zenith[0] + zenith[1] * zenith[1] + zenith[2] * zenith[2]).sqrt();
            assert!((n - 1.0).abs() < 1e-15, "zenith is not a unit vector: {n}");
            for k in 0..3 {
                assert_eq!(centre[k], zenith[k] * RE_EARTH, "centre is not R⊕ · zenith");
            }
        }

        // Degrees, not radians: sin(45°) = 0.7071, sin(45 rad) = 0.8509.
        let sin_lat = cells[2].1[2];
        assert!(
            (sin_lat - FRAC_1_SQRT_2).abs() < 1e-15,
            "latitude was not converted from degrees: sinφ = {sin_lat}, and sin(45 rad) = {}",
            45.0_f64.sin()
        );
    }

    /// Range, range rate, elevation and azimuth, each against the geometry that defines it, at a
    /// station away from the equator so the topocentric frame is not the identity and the latitude
    /// argument's units are observable.
    #[test]
    fn the_measurement_models_are_the_geometry_they_claim() {
        let station = geodetic_to_itrf(45.0, 0.0, 0.0);

        // Radially out from an equatorial station, so the range is a subtraction and the range rate
        // is the x-component of the velocity — the other two are transverse and must not appear.
        let eq = [RE_EARTH, 0.0, 0.0];
        let sat = [RE_EARTH + 500e3, 0.0, 0.0, 1000.0, 2000.0, 3000.0];
        assert_eq!(range(sat, eq), 500e3);
        assert_eq!(range_rate(sat, eq), 1000.0);

        // Straight up the *geodetic* zenith, which at 45° is not the geocentric radial: only a
        // station frame built from the ellipsoid normal puts this at the local vertical.
        let overhead = geodetic_to_itrf(45.0, 0.0, 500e3);
        let up = [overhead[0], overhead[1], overhead[2], 0.0, 0.0, 0.0];
        let el = elevation(up, station, 45.0_f64.to_radians(), 0.0);
        assert!(
            (el - FRAC_PI_2).abs() < 1e-9,
            "a satellite on the geodetic zenith is at {el} rad, not π/2 — the latitude argument is \
             radians, and the frame is the ellipsoid normal"
        );

        // Azimuth runs from north through east: displacing the same satellite north must give 0 and
        // east must give +π/2. The opposite convention swaps them.
        let north = [
            overhead[0] - 50e3,
            overhead[1],
            overhead[2] + 50e3,
            0.0,
            0.0,
            0.0,
        ];
        let east = [overhead[0], overhead[1] + 50e3, overhead[2], 0.0, 0.0, 0.0];
        let (az_n, az_e) = (
            azimuth(north, station, 45.0_f64.to_radians(), 0.0),
            azimuth(east, station, 45.0_f64.to_radians(), 0.0),
        );
        assert!(
            az_n.abs() < 1e-9,
            "a northward displacement bears {az_n} rad, not 0"
        );
        assert!(
            (az_e - FRAC_PI_2).abs() < 1e-9,
            "an eastward displacement bears {az_e} rad, not π/2"
        );

        // Below the horizon is negative, not wrapped or clamped.
        let antipode = [-RE_EARTH - 500e3, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(elevation(antipode, eq, 0.0, 0.0) < 0.0);
    }

    /// The five points in order, each checked against its own defining equation rather than against
    /// a baked coordinate: the equilateral pair in closed form, the three collinear ones by the
    /// residual of the synodic equilibrium `x − (1−μ)(x+μ)/|x+μ|³ − μ(x−1+μ)/|x−1+μ|³`.
    #[test]
    fn the_libration_points_come_back_in_order_and_satisfy_their_own_equilibrium() {
        let mu = 0.012_150_584_460_351;
        let l = libration_points(mu);

        for (i, p) in l.iter().enumerate() {
            assert_eq!(p[2], 0.0, "L{} left the synodic plane", i + 1);
        }
        for p in &l[..3] {
            assert_eq!(p[1], 0.0, "a collinear point is off the x-axis: {p:?}");
        }

        // L1 between the primaries, L2 beyond the secondary, L3 on the far side of the primary.
        assert!(
            -mu < l[0][0] && l[0][0] < 1.0 - mu,
            "L1 is not between them: {:?}",
            l[0]
        );
        assert!(
            l[1][0] > 1.0 - mu,
            "L2 is not beyond the secondary: {:?}",
            l[1]
        );
        assert!(l[2][0] < -mu, "L3 is not beyond the primary: {:?}", l[2]);

        let residual = |x: f64| {
            let (r1, r2) = ((x + mu).abs(), (x - 1.0 + mu).abs());
            x - (1.0 - mu) * (x + mu) / (r1 * r1 * r1) - mu * (x - 1.0 + mu) / (r2 * r2 * r2)
        };
        for (i, p) in l[..3].iter().enumerate() {
            let e = residual(p[0]).abs();
            assert!(
                e < 1e-12,
                "L{} is not an equilibrium: residual {e:.3e}",
                i + 1
            );
        }

        // The equilateral pair, exactly: x = ½ − μ in a barycentric frame, y = ±√3/2, L4 leading.
        assert_eq!(l[3][0], 0.5 - mu);
        assert_eq!(l[4][0], 0.5 - mu);
        assert!(
            (l[3][1] - 3.0_f64.sqrt() / 2.0).abs() < 1e-15,
            "L4 y = {}",
            l[3][1]
        );
        assert_eq!(l[4][1], -l[3][1], "L5 is not L4 reflected");
    }

    /// The geodetic conversion reads degrees and the WGS-84 ellipsoid, not radians and a sphere.
    /// Both mistakes are silent on the one input every fixture uses — the origin — where the two
    /// agree exactly.
    #[test]
    fn the_geodetic_conversion_reads_degrees_and_the_ellipsoid() {
        assert_eq!(geodetic_to_itrf(0.0, 0.0, 0.0), [RE_EARTH, 0.0, 0.0]);
        assert_eq!(
            geodetic_to_itrf(0.0, 0.0, 1000.0),
            [RE_EARTH + 1000.0, 0.0, 0.0]
        );

        // The pole is the semi-minor axis, 21 km below the equatorial radius a sphere would give.
        let pole = geodetic_to_itrf(90.0, 0.0, 0.0);
        assert!(
            pole[0].abs() < 1e-6 && pole[1].abs() < 1e-6,
            "the pole is off-axis: {pole:?}"
        );
        assert!(
            (pole[2] - WGS84_B).abs() < 1e-3,
            "the polar radius is {} m, not the WGS-84 semi-minor axis {WGS84_B}",
            pole[2]
        );

        // 90° east is +y, which a radian reading (90 rad ≡ 1.99 rad ≈ 114°) would put in the second
        // quadrant instead.
        let east = geodetic_to_itrf(0.0, 90.0, 0.0);
        assert!(
            east[0].abs() < 1e-6 && east[1] > 6.3e6,
            "90° east is {east:?}"
        );

        // A quarter turn of longitude is a rotation, so the norm cannot move.
        let n = |p: [f64; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!((n(east) - RE_EARTH).abs() < 1e-6);
    }

    /// Two-body propagation across the bridge: the identity at `dt = 0`, and reversible, which is
    /// what a dropped or mis-signed `dt` breaks.
    #[test]
    fn kepler_propagation_is_the_identity_at_zero_and_reverses() {
        let r0 = [7_000e3, 0.0, 0.0];
        let v0 = [0.0, 7_546.0, 0.0];
        assert_eq!(kepler_propagate(r0, v0, MU_EARTH, 0.0), (r0, v0));

        let (r1, v1) = kepler_propagate(r0, v0, MU_EARTH, 1_800.0);
        assert!(
            (r1[0] - r0[0]).abs() > 1e6,
            "half an hour did not move the satellite: {r1:?}"
        );
        let (r2, v2) = kepler_propagate(r1, v1, MU_EARTH, -1_800.0);
        for k in 0..3 {
            assert!(
                (r2[k] - r0[k]).abs() < 1e-3,
                "position did not reverse: {r2:?} vs {r0:?}"
            );
            assert!(
                (v2[k] - v0[k]).abs() < 1e-6,
                "velocity did not reverse: {v2:?} vs {v0:?}"
            );
        }
    }

    /// A real TU Delft CHAMP excerpt, thirty accelerometer rows, shared by the two density bridges
    /// below. Every row in it is nominal with a positive orbit mean, which is why the flag test
    /// appends its own rows rather than looking for one here.
    const CHAMP: &str =
        include_str!("../../spacelab-propagation/tests/fixtures/champ_density_sample.txt");

    /// A real CelesTrak `SW-All.csv` excerpt: three 1957 rows and four spanning the 2000-07-29 CHAMP
    /// window, so the space-weather bridge is read against the same file the parser is.
    const CELESTRAK: &str =
        include_str!("../../spacelab-propagation/tests/fixtures/celestrak_sw_sample.csv");

    /// The five columns Python hands in — `[passes, coverage_fraction, max_gap, mean_gap,
    /// mean_response_time]` — become five *named* fields of `CoverageMetrics`, and the eight the
    /// aggregate returns come back as a positional tuple. Both mappings are this file's, and both
    /// are silent when wrong: a Python caller who gets `mean_max_gap` where `mean_response_time`
    /// belongs receives a plausible number in seconds.
    ///
    /// So every input column carries a different value and every output field a different one, and
    /// the weighted case is run against the unweighted: `mean_coverage` `0.8125` vs `0.625` is what
    /// says the weights reached `constellation_fom` rather than being dropped. All eight expected
    /// values are dyadic rationals, so the comparison is exact rather than tolerant.
    #[test]
    fn the_constellation_fom_bridge_maps_five_named_columns_onto_a_positional_tuple() {
        let points = vec![
            [2.0, 0.25, 900.0, 111.0, 300.0],
            [4.0, 1.0, 500.0, 222.0, 100.0],
        ];

        // total weight 4: mean_coverage (0.25 + 3)/4, mean_max_gap (900 + 1500)/4,
        // mean_response_time (300 + 300)/4. Worst-case and counts ignore the weights.
        assert_eq!(
            constellation_fom(points.clone(), Some(vec![1.0, 3.0])),
            (2, 0.8125, 0.25, 900.0, 600.0, 150.0, 300.0, 1)
        );

        // The same points weighted equally. Only the three means move, and they must all move.
        assert_eq!(
            constellation_fom(points, None),
            (2, 0.625, 0.25, 900.0, 700.0, 200.0, 300.0, 1)
        );

        // `mean_gap` — column 3 — is the one input the figures of merit do not consume, so it is
        // also the one a transposition would hide behind. Changing it alone must change nothing.
        let shifted = vec![
            [2.0, 0.25, 900.0, -5.0, 300.0],
            [4.0, 1.0, 500.0, 1e9, 100.0],
        ];
        assert_eq!(
            constellation_fom(shifted, None),
            (2, 0.625, 0.25, 900.0, 700.0, 200.0, 300.0, 1)
        );
    }

    /// Nine geometry fields flattened into one positional tuple, seven of them angles in radians of
    /// the same order of magnitude — the shape where a reordering is invisible to any assertion
    /// weaker than the values themselves.
    ///
    /// The reference is the crate's own [`StreetOfCoverage::geometry`], field by field and bit for
    /// bit; its values are gated against `derivations/soc_street_geometry.mac` in `soc.rs`, so this
    /// test pins only what the bridge adds: the order of the arguments and of the nine fields. 87°
    /// and 90° differ only through the inclination, so an argument swap would not reproduce both.
    /// 80° at 10° elevation is refused: its street half-width is below `π/2 − i`, where the
    /// counter-rotating seam cannot close.
    #[test]
    fn the_street_of_coverage_bridge_returns_nine_geometry_fields_in_the_documented_order() {
        let eps = 10.0_f64.to_radians();
        for inc_deg in [87.0_f64, 90.0] {
            let inc = inc_deg.to_radians();
            let got = street_of_coverage_geometry(inc, 800e3, eps, 6_378_137.0, MU_EARTH, None)
                .expect("a 800 km street at 10 degrees elevation is feasible at 87 and 90 degrees");
            let want = StreetOfCoverage {
                inclination: Angle::from_radians(inc),
                altitude: 800e3,
                min_elevation: Angle::from_radians(eps),
                earth_radius: 6_378_137.0,
                mu: MU_EARTH,
                per_plane: None,
            }
            .geometry()
            .expect("the same street through the crate");
            assert_eq!(
                (got.0, got.1),
                (want.planes, want.per_plane),
                "{inc_deg}°: counts"
            );
            for (got, want, name) in [
                (got.2, want.footprint_half_angle, "footprint_half_angle"),
                (got.3, want.street_half_width, "street_half_width"),
                (got.4, want.raan_step_co, "raan_step_co"),
                (got.5, want.raan_step_counter, "raan_step_counter"),
                (got.6, want.raan_step, "raan_step"),
                (got.7, want.raan_span, "raan_span"),
                (got.8, want.inter_plane_phase, "inter_plane_phase"),
            ] {
                assert_eq!(got.to_bits(), want.to_bits(), "{inc_deg}°: {name}");
            }
        }

        let inc = 80.0_f64.to_radians();
        assert!(
            street_of_coverage_geometry(inc, 800e3, eps, 6_378_137.0, MU_EARTH, None).is_none(),
            "an 80-degree street of half-width 6 degrees cannot close its counter-rotating seam"
        );
        assert!(
            street_of_coverage_geometry(FRAC_PI_2, 800e3, eps, 6_378_137.0, MU_EARTH, Some(2))
                .is_none(),
            "two satellites per plane are below the pi/theta continuity floor and must be refused"
        );
    }

    /// Four columns that must stay aligned row-for-row, because Python zips them back together: a
    /// bridge that collected latitude from one filtered pass and `log_ratio` from another would put
    /// each ratio at the wrong place on the globe and the arrays would still be the same length.
    ///
    /// The expected values were recomputed from the fixture text — `ln(rho / rho_mean)` and
    /// `radians(latitude)` read straight off columns 8, 9 and 5 — rather than read back from this
    /// function. The summed `log_ratio` pins all thirty at once; the two endpoints pin the order.
    #[test]
    fn the_density_sample_bridge_returns_four_row_aligned_columns() {
        let (lat, lst, log_ratio, weight) = density_data_samples(CHAMP);
        assert_eq!(lat.len(), 30, "the excerpt is thirty rows");
        assert_eq!((lst.len(), log_ratio.len(), weight.len()), (30, 30, 30));

        assert!((lat[0] - 0.696_013_020_660_527_5).abs() < 1e-15);
        assert!((lst[0] - 0.963).abs() < 1e-15);
        assert!((log_ratio[0] + 0.514_139_054_418_471_3).abs() < 1e-15);

        assert!((lat[29] - 0.372_039_778_167_829_24).abs() < 1e-15);
        assert!((lst[29] - 1.043).abs() < 1e-15);
        assert!((log_ratio[29] + 0.331_569_745_476_636_37).abs() < 1e-15);

        let total: f64 = log_ratio.iter().sum();
        assert!(
            (total + 13.317_832_673_905_718).abs() < 1e-12,
            "the thirty ratios do not sum to the file's: {total}"
        );
        assert!(
            weight.iter().all(|w| *w == 1.0),
            "unweighted by construction"
        );
    }

    /// The absolute-amplitude sibling of the sample bridge, and the one whose filter is written
    /// here rather than in the core: `o.nominal && o.density > 0.0` is this file's line, so the
    /// excerpt — thirty rows, all nominal, all positive — cannot exercise it on its own. Two
    /// appended rows do, one flagged and one at zero density, and both must vanish.
    ///
    /// `ln rho_ref = ln rho + (altitude - reference) / scale_height` was recomputed from the
    /// fixture's columns 8 and 3. The reference sits 61-71 km *below* the sample, so at a 50 km
    /// scale height the correction is `+1.42`, not a rounding term: a dropped or inverted altitude
    /// offset moves the answer in its second figure, not its last.
    #[test]
    fn the_height_normalized_density_bridge_drops_flagged_and_non_positive_rows() {
        let text = format!(
            "{CHAMP}\n\
2000-07-29 02:00:00.000 GPS 470000.000 1.0 45.0 12.0 140.0 0.10000000E-11 0.20000000E-11 1.0 0.0\n\
2000-07-29 02:01:00.000 GPS 470000.000 1.0 45.0 12.0 140.0 0.00000000E+00 0.20000000E-11 0.0 0.0\n"
        );

        let (years, months, days, lat, lst, ln_rho) =
            density_altitude_normalized(&text, 400_000.0, 50_000.0);
        assert_eq!(
            ln_rho.len(),
            30,
            "the flagged row, the zero-density row, or both survived the filter"
        );
        assert_eq!((years.len(), months.len(), days.len()), (30, 30, 30));
        assert_eq!((lat.len(), lst.len()), (30, 30));

        assert!(years.iter().all(|y| *y == 2000) && months.iter().all(|m| *m == 7));
        assert!(days.iter().all(|d| *d == 29), "the excerpt is one day");

        assert!((lat[0] - 0.696_013_020_660_527_5).abs() < 1e-15);
        assert!((lst[0] - 0.963).abs() < 1e-15);
        assert!((ln_rho[0] + 26.055_193_418_065_17).abs() < 1e-12);
        assert!((ln_rho[29] + 26.079_413_214_202_763).abs() < 1e-12);

        // Halving the scale height doubles the altitude correction, so the difference between the
        // two runs is the correction itself — (471231.023 - 400000) / 50000 — and that is what
        // separates the third argument from the second without needing a second file.
        let (.., halved) = density_altitude_normalized(&text, 400_000.0, 25_000.0);
        assert!(
            (halved[0] - ln_rho[0] - 1.424_620_459_999_999_6).abs() < 1e-12,
            "the scale height is not dividing the altitude offset: {} vs {}",
            halved[0],
            ln_rho[0]
        );
    }

    /// Six aligned columns off a real CelesTrak file. The dates are the discriminator: the excerpt
    /// deliberately holds three 1957 rows and four from 2000, so a bridge that returned the year
    /// where the month belongs, or that reordered the rows, cannot reproduce the sequence. `ap`,
    /// `f107_observed` and `f107_center81` are columns 20, 24 and 27 of the CSV and are all in the
    /// same range, which is exactly why each is checked against its own file value.
    #[test]
    fn the_space_weather_bridge_returns_six_row_aligned_columns() {
        let (years, months, days, ap, f107, f107c81) = space_weather_days(CELESTRAK);
        assert_eq!(years.len(), 7, "the excerpt is seven days");
        assert_eq!((months.len(), days.len()), (7, 7));
        assert_eq!((ap.len(), f107.len(), f107c81.len()), (7, 7, 7));

        assert_eq!(years, vec![1957, 1957, 1957, 2000, 2000, 2000, 2000]);
        assert_eq!(months, vec![10, 10, 10, 7, 7, 7, 8]);
        assert_eq!(days, vec![1, 2, 3, 29, 30, 31, 1]);

        assert_eq!(ap, vec![21.0, 12.0, 19.0, 27.0, 8.0, 21.0, 12.0]);
        assert_eq!(f107, vec![269.3, 253.3, 266.3, 153.2, 149.9, 147.9, 149.4]);
        assert_eq!(
            f107c81,
            vec![266.6, 267.4, 268.1, 181.1, 181.0, 180.6, 180.0]
        );
    }

    /// The seven-element storm-time `ap_array` off the same file. Its layout is documented --
    /// `[daily Ap, current 3 h, -3 h, -6 h, -9 h, mean(-12..-33 h), mean(-36..-57 h)]` -- so slot 0
    /// has to be the daily `Ap` the column read gives, and the whole array is `None` for a day whose
    /// ~2.5-day look-back is not in the file. Two dates in the excerpt separate those cases.
    #[test]
    fn the_ap_array_bridge_needs_its_look_back_and_leads_with_the_daily_ap() {
        let (.., ap, _, _) = space_weather_days(CELESTRAK);
        let array = space_weather_ap_array(CELESTRAK, 2000, 8, 1, 43_200.0)
            .expect("2000-08-01 has three preceding days in the excerpt");
        assert_eq!(array.len(), 7, "the NRLMSISE-00 ap_array is seven elements");
        assert!(
            (array[0] - ap[6]).abs() < 1e-12,
            "slot 0 is {} but 2000-08-01's daily Ap is {}",
            array[0],
            ap[6]
        );
        for (k, &a) in array.iter().enumerate() {
            assert!((0.0..400.0).contains(&a), "slot {k} holds {a}, not an ap");
        }

        assert!(
            space_weather_ap_array(CELESTRAK, 1957, 10, 1, 0.0).is_none(),
            "the first day in the file has no look-back"
        );
        assert!(
            space_weather_ap_array(CELESTRAK, 1999, 1, 1, 0.0).is_none(),
            "a day the file does not carry has no array"
        );
    }

    /// The Step-2 solid-tide corrections are a five-element record, so the oracle is the crate
    /// function under the same arguments and the discriminator is that the five entries are
    /// genuinely different numbers at the sub-nanometre scale the series works at -- which no
    /// constant array reproduces.
    #[test]
    fn the_step_two_tide_corrections_are_five_distinct_terms() {
        // Delaunay arguments and GMST at an arbitrary but fixed epoch.
        let gmst = 1.234;
        let delaunay = [2.355, 6.226, 3.393, 5.198, 1.146];
        let got = solid_tide_step2(gmst, delaunay);
        assert_eq!(got, spacelab_propagation::step2_corrections(gmst, delaunay));

        // Step 2 is the ~mm frequency dependence on top of the Step-1 Love numbers: 1e-14..1e-9.
        for (k, &c) in got.iter().enumerate() {
            assert!(
                (1e-14..1e-9).contains(&c.abs()),
                "term {k} is {c}, outside the frequency-dependent scale"
            );
        }
        for i in 0..5 {
            for j in (i + 1)..5 {
                assert!(
                    (got[i] - got[j]).abs() > 1e-15,
                    "terms {i} and {j} are the same number: {}",
                    got[i]
                );
            }
        }
        // The series is driven by the arguments, so a different epoch is a different correction.
        let moved = solid_tide_step2(gmst + 1.0, delaunay);
        assert!(
            (moved[1] - got[1]).abs() > 1e-15,
            "GMST does not reach ΔC21"
        );
    }

    /// The five DOPs against the two identities that define them — `GDOP² = PDOP² + TDOP²` and
    /// `PDOP² = HDOP² + VDOP²` — which is what a positional tuple has to be checked against, because
    /// the tuple carries no field names across the boundary. The identities are not enough on their
    /// own: they survive swapping HDOP with VDOP. What separates those two is the geometry itself —
    /// a receiver that sees satellites only above its horizon is always worse in the vertical, so
    /// `VDOP > HDOP` here is a property of the sky, not of this implementation.
    #[test]
    fn the_dop_tuple_is_the_five_dilutions_in_their_defining_order() {
        // Local ENU at (0°, 0°) is (up, east, north) = (x̂, ŷ, ẑ), so a line of sight at elevation
        // `el` and azimuth `az` is (sin el, cos el·sin az, cos el·cos az). One high satellite plus a
        // 120°-spaced ring at 30°: well conditioned, and asymmetric enough that no two DOPs coincide.
        let receiver = [RE_EARTH, 0.0, 0.0];
        let los = |el_deg: f64, az_deg: f64| {
            let (el, az) = (el_deg.to_radians(), az_deg.to_radians());
            [
                RE_EARTH + 20_200e3 * el.sin(),
                20_200e3 * el.cos() * az.sin(),
                20_200e3 * el.cos() * az.cos(),
            ]
        };
        let sats = vec![
            los(80.0, 0.0),
            los(30.0, 0.0),
            los(30.0, 120.0),
            los(30.0, 240.0),
        ];

        let (gdop, pdop, hdop, vdop, tdop) =
            dilution_of_precision(receiver, sats.clone()).expect("four satellites is not singular");

        assert!(
            (gdop * gdop - (pdop * pdop + tdop * tdop)).abs() < 1e-9,
            "GDOP² ≠ PDOP² + TDOP²: {gdop}, {pdop}, {tdop}"
        );
        assert!(
            (pdop * pdop - (hdop * hdop + vdop * vdop)).abs() < 1e-9,
            "PDOP² ≠ HDOP² + VDOP²: {pdop}, {hdop}, {vdop}"
        );
        assert!(
            vdop > hdop,
            "the vertical is not the worse axis: {vdop} vs {hdop}"
        );
        for d in [gdop, pdop, hdop, vdop, tdop] {
            assert!(d > 0.0 && d < 100.0, "DOP out of range: {d}");
        }

        // Four unknowns (three position, one clock) need four rows.
        assert!(dilution_of_precision(receiver, sats[..3].to_vec()).is_none());
        assert!(dilution_of_precision(receiver, vec![]).is_none());
    }

    /// The swath tuple against the spherical geometry that produced it: `ground_range = Rₑ·λ`, the
    /// law of sines `(Rₑ+h)·sin η = Rₑ·cos ε`, and the law of cosines for the slant range. Each
    /// identity ties a different pair of slots, so no permutation of the four satisfies all three.
    #[test]
    fn the_swath_tuple_is_the_spherical_geometry_in_the_documented_order() {
        let (re, alt) = (RE_EARTH, 700e3);

        let (lambda, elev, ground, slant) =
            swath_geometry(re, alt, 30f64.to_radians()).expect("30° is well inside the limb");
        assert!(
            (ground - re * lambda).abs() < 1e-6,
            "ground range is not Rₑ·λ"
        );
        assert!(
            ((re + alt) * 30f64.to_radians().sin() - re * elev.cos()).abs() < 1e-6,
            "the look/elevation pair does not close the law of sines"
        );
        let by_cosines =
            (re * re + (re + alt) * (re + alt) - 2.0 * re * (re + alt) * lambda.cos()).sqrt();
        assert!(
            (slant - by_cosines).abs() < 1e-6,
            "slant range is not the third side"
        );

        // Nadir is the degenerate case that tells the four slots apart by value: only the slant
        // range is non-zero-ish, and only the elevation is a right angle.
        let (lambda0, elev0, ground0, slant0) = swath_geometry(re, alt, 0.0).unwrap();
        assert_eq!((lambda0, ground0), (0.0, 0.0));
        assert!((elev0 - FRAC_PI_2).abs() < 1e-12);
        assert!((slant0 - alt).abs() < 1e-6);

        // Past the limb the line of sight misses the Earth entirely.
        let limb = horizon_look_angle(re, alt);
        assert!(
            swath_geometry(re, alt, limb * 1.001).is_none(),
            "the limb is not enforced"
        );
        assert!(
            swath_geometry(re, alt, limb).is_some(),
            "the limb itself still grazes"
        );
    }

    /// Anchors are laid row-major over the feature map at pixel centres, and each is the scale/ratio
    /// parameterisation `w = s/√r`, `h = s·√r` — so the area is `s²` and the aspect is `r`,
    /// independently of where the box sits. The loop order is the part a bridge can silently
    /// transpose, so the first two cells are checked by value.
    #[test]
    fn the_anchor_boxes_are_row_major_pixel_centres_of_the_right_area_and_aspect() {
        let (stride, scales, ratios) = (16.0, vec![32.0, 64.0], vec![0.5, 2.0]);
        let anchors = generate_anchors(2, 3, stride, scales.clone(), ratios.clone());
        assert_eq!(
            anchors.len(),
            2 * 3 * 2 * 2,
            "one anchor per cell × scale × ratio"
        );

        for (i, &(x1, y1, x2, y2)) in anchors.iter().enumerate() {
            let (w, h) = (x2 - x1, y2 - y1);
            let scale = scales[(i / ratios.len()) % scales.len()];
            let ratio = ratios[i % ratios.len()];
            assert!(
                (w * h - scale * scale).abs() < 1e-9,
                "anchor {i} has area ≠ s²"
            );
            assert!(
                (h / w - ratio).abs() < 1e-12,
                "anchor {i} has the wrong aspect"
            );

            // Centres are at (col + ½)·stride, columns varying fastest.
            let cell = i / (scales.len() * ratios.len());
            let (row, col) = (cell / 3, cell % 3);
            assert!((0.5 * (x1 + x2) - (col as f64 + 0.5) * stride).abs() < 1e-12);
            assert!((0.5 * (y1 + y2) - (row as f64 + 0.5) * stride).abs() < 1e-12);
        }
    }

    /// The Faster-R-CNN decode is exact arithmetic, so it is checked by value rather than by
    /// property: `cx + dx·w`, `cy + dy·h`, `w·e^{dw}`, `h·e^{dh}`, then back to corner form. The
    /// anchor is non-square and off-origin so a swapped axis or a dropped centre shift shows.
    #[test]
    fn the_box_decode_is_the_centre_form_arithmetic() {
        let anchor = (10.0, 20.0, 30.0, 60.0); // centre (20, 40), size 20 × 40
        assert_eq!(
            decode_box(anchor, (0.0, 0.0, 0.0, 0.0)),
            anchor,
            "a null delta must be identity"
        );

        let (x1, y1, x2, y2) = decode_box(
            anchor,
            (
                0.5,
                -0.25,
                core::f64::consts::LN_2,
                -core::f64::consts::LN_2,
            ),
        );
        // cx = 20 + 0.5·20 = 30, cy = 40 − 0.25·40 = 30, w = 20·2 = 40, h = 40/2 = 20.
        for (got, want) in [(x1, 10.0), (y1, 20.0), (x2, 50.0), (y2, 40.0)] {
            assert!((got - want).abs() < 1e-12, "decoded {got}, expected {want}");
        }
    }

    /// Sentman's four coefficients are `(pressure, shear, drag, lift)`, and the last two are the
    /// first two rotated into the flow frame: `drag = p·cos δ + τ·sin δ`, `lift = p·sin δ − τ·cos δ`.
    /// At normal incidence the rotation is the identity and the tangential pair vanishes, which
    /// pins all four slots without needing a reference value for any of them.
    #[test]
    fn the_sentman_tuple_is_pressure_shear_drag_lift() {
        let (s, alpha, t_wall, t_atm) = (7.5, 0.93, 300.0, 1000.0);

        let cos_d = 0.6_f64;
        let sin_d = (1.0 - cos_d * cos_d).sqrt();
        let (p, tau, drag, lift) = sentman_drag_coefficients(s, cos_d, alpha, t_wall, t_atm);
        assert!(
            (drag - (p * cos_d + tau * sin_d)).abs() < 1e-12,
            "drag is not the rotated pair"
        );
        assert!(
            (lift - (p * sin_d - tau * cos_d)).abs() < 1e-12,
            "lift is not the rotated pair"
        );
        assert!(
            p > 0.0 && tau > 0.0,
            "an inclined windward panel has both: {p}, {tau}"
        );

        // Normal incidence: no tangential momentum, so shear and lift are identically zero and the
        // drag collapses onto the pressure.
        let (p0, tau0, drag0, lift0) = sentman_drag_coefficients(s, 1.0, alpha, t_wall, t_atm);
        assert_eq!(tau0, 0.0, "a normal panel cannot carry shear");
        assert_eq!(lift0, 0.0, "a normal panel cannot carry lift");
        assert!(
            (drag0 - p0).abs() < 1e-12,
            "at δ = 0 the drag is the pressure"
        );
        assert!(p0 > 2.0 && p0 < 3.0, "hyperthermal Cp is ~2–3, got {p0}");
    }

    /// Draim's geometry is Kepler's third law plus the conic: `a = (μ(T/2π)²)^{1/3}`, `r_p = a(1−e)`,
    /// `r_a = a(1+e)`. All three are computed here from the inputs alone.
    #[test]
    fn the_draim_geometry_is_the_period_and_the_conic() {
        let (period, ecc, mu) = (172_800.0, 0.28, 3.986_004_418e14);
        let (a, rp, ra) = draim_geometry(period, ecc, 0.575_958_653_158_128_8, mu);

        let a_kepler = (mu * (period / core::f64::consts::TAU).powi(2)).cbrt();
        assert!(
            (a / a_kepler - 1.0).abs() < 1e-12,
            "a = {a}, Kepler gives {a_kepler}"
        );
        assert!((rp - a * (1.0 - ecc)).abs() < 1e-6, "perigee is not a(1−e)");
        assert!((ra - a * (1.0 + ecc)).abs() < 1e-6, "apogee is not a(1+e)");
        assert!(
            rp < a && a < ra,
            "the three radii are out of order: {rp}, {a}, {ra}"
        );
    }

    /// A projection round trip: two pinhole cameras see one world point, and triangulating their two
    /// pixels has to return it. Constructed rather than tabulated, so the assertion is on the
    /// bridge's marshalling — the camera tuple in, the world tuple out — and not on a stored number.
    #[test]
    fn triangulating_two_projections_returns_the_world_point() {
        let cam_a = PyCamera::new(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            1000.0,
            [0.0, 0.0],
        );
        let cam_b = PyCamera::new(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [10.0, -3.0, 0.0],
            1000.0,
            [0.0, 0.0],
        );
        let world = (1.0, 2.0, 20.0);

        let (pa, pb) = (cam_a.project(world), cam_b.project(world));
        let got = triangulate(vec![cam_a, cam_b], vec![pa, pb])
            .expect("two distinct centres are not degenerate");
        for (g, w) in [(got.0, world.0), (got.1, world.1), (got.2, world.2)] {
            assert!((g - w).abs() < 1e-9, "triangulated {g}, expected {w}");
        }

        // One view cannot fix a depth.
        assert!(triangulate(vec![cam_a], vec![pa]).is_none());

        // The same point through the track builder: one match across two views is one track.
        let tracks = sfm_triangulate(
            vec![vec![(pa.0, pa.1, 1.0)], vec![(pb.0, pb.1, 1.0)]],
            vec![(0, 0, 1, 0)],
            vec![cam_a, cam_b],
        );
        assert_eq!(tracks.len(), 1, "one match is one track");
        let t = tracks[0].expect("the track's two observations triangulate");
        for (g, w) in [(t.0, world.0), (t.1, world.1), (t.2, world.2)] {
            assert!((g - w).abs() < 1e-9, "track gave {g}, expected {w}");
        }
    }

    /// The Pauli powers are `[odd, even, volume]`, and each of the three canonical scatterers puts
    /// all the power in exactly one of them: a sphere is odd-bounce, a dihedral even, and a
    /// cross-polarising target volume. Three cases pin three slots, and the span-power identity
    /// `Σ = |S_hh|² + 2|S_hv|² + |S_vv|²` says nothing was dropped on the way through.
    #[test]
    fn the_pauli_powers_separate_the_three_canonical_scatterers() {
        let zero = (0.0, 0.0);
        let (odd, even, vol) = pauli_powers((1.0, 0.0), zero, (1.0, 0.0));
        assert!(
            (odd - 2.0).abs() < 1e-12 && even.abs() < 1e-12 && vol.abs() < 1e-12,
            "sphere: {odd}, {even}, {vol}"
        );

        let (odd, even, vol) = pauli_powers((1.0, 0.0), zero, (-1.0, 0.0));
        assert!(
            odd.abs() < 1e-12 && (even - 2.0).abs() < 1e-12 && vol.abs() < 1e-12,
            "dihedral: {odd}, {even}, {vol}"
        );

        let (odd, even, vol) = pauli_powers(zero, (1.0, 0.0), zero);
        assert!(
            odd.abs() < 1e-12 && even.abs() < 1e-12 && (vol - 2.0).abs() < 1e-12,
            "volume: {odd}, {even}, {vol}"
        );

        // A general complex matrix: the three powers still sum to the span.
        let (hh, hv, vv) = ((0.8, -0.3), (0.1, 0.25), (-0.5, 0.4));
        let (odd, even, vol) = pauli_powers(hh, hv, vv);
        let span = hh.0 * hh.0
            + hh.1 * hh.1
            + 2.0 * (hv.0 * hv.0 + hv.1 * hv.1)
            + vv.0 * vv.0
            + vv.1 * vv.1;
        assert!(
            (odd + even + vol - span).abs() < 1e-12,
            "the decomposition lost power: {} vs {span}",
            odd + even + vol
        );
    }

    /// The zonal drift is linear in time, so the correction is zero at J2000 and the conventional
    /// coefficients are the mean values plus exactly that correction. `C̄₂₀ ≈ −J₂/√5 = −4.84e−4` is
    /// the one absolute number here, and it is what says the bridge returned normalised
    /// coefficients in degree order rather than three interchangeable small numbers.
    #[test]
    fn the_zonal_drift_is_linear_and_rides_on_the_mean_field() {
        assert_eq!(time_variable_zonal_correction(0.0), (0.0, 0.0, 0.0));

        let (d20, d30, d40) = time_variable_zonal_correction(10.0);
        let (r20, r30, r40) = time_variable_zonal_correction(1.0);
        for (decade, rate) in [(d20, r20), (d30, r30), (d40, r40)] {
            assert!(rate != 0.0, "a zero drift rate is not a drift");
            assert!(
                (decade - 10.0 * rate).abs() < 1e-20,
                "the drift is not linear in time"
            );
        }

        let (c20, c30, c40) = conventional_zonal_coefficients(0.0);
        assert!((c20 + 4.841e-4).abs() < 1e-6, "C̄₂₀ is not −J₂/√5: {c20}");
        assert!(
            c20.abs() > 100.0 * c30.abs(),
            "C̄₂₀ and C̄₃₀ are the wrong way round"
        );
        assert!(
            c30.abs() < 1e-5 && c40.abs() < 1e-5,
            "C̄₃₀/C̄₄₀ are out of range: {c30}, {c40}"
        );

        // The tolerance is relative to the drift, not absolute. A decade of C̄₂₀ drift is 1.2e−10 on
        // a mean of −4.8e−4, so `then − now` cancels five significant digits and cannot resolve
        // better than ~1e−19 whatever the code does; an absolute 1e−20 would be testing f64.
        let (t20, t30, t40) = conventional_zonal_coefficients(10.0);
        for (then, now, drift) in [(t20, c20, d20), (t30, c30, d30), (t40, c40, d40)] {
            assert!(
                (then - now - drift).abs() < 1e-6 * drift.abs(),
                "conventional ≠ mean + secular: {then} − {now} ≠ {drift}"
            );
        }
    }

    /// IGRF returns the field in ECEF, in tesla. Two things are checked against geomagnetism rather
    /// than against a stored vector: the surface field is tens of microtesla, and in the northern
    /// hemisphere the field points *into* the ground, so its radial component is negative.
    #[test]
    fn the_geomagnetic_field_is_ecef_tesla_pointing_into_the_northern_ground() {
        let (lat, lon) = (45f64.to_radians(), 10f64.to_radians());
        let r = geodetic_to_itrf(45.0, 10.0, 0.0);
        let (bx, by, bz) = igrf_field_ecef(r[0], r[1], r[2], 2026.5);

        let magnitude = (bx * bx + by * by + bz * bz).sqrt();
        assert!(
            (20e-6..70e-6).contains(&magnitude),
            "|B| = {magnitude} T is not a surface field in tesla"
        );

        let up = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
        let radial = bx * up[0] + by * up[1] + bz * up[2];
        assert!(
            radial < 0.0,
            "the northern field does not dip into the ground: {radial}"
        );
    }

    /// The coverage bridge, end to end on the real access engine: a satellite in a 500 km equatorial
    /// orbit over an equatorial cell and a 60°N cell. The five columns are `[passes,
    /// coverage_fraction, max_gap, mean_gap, mean_response]`, and the discriminator is the pair of
    /// cells — the equatorial one is seen, the high-latitude one cannot be, so a bridge that
    /// returned a constant row, or that transposed the two cells, disagrees with the geometry.
    #[test]
    fn the_coverage_and_access_bridges_run_the_real_engine_over_two_cells() {
        let sensor = Sensor {
            kind: SensorKind::Elevation(
                ElevationMask::new(5f64.to_radians()).expect("5° is a valid mask"),
            ),
        };
        let a = RE_EARTH + 500e3;
        let cells = grid_cells(vec![0.0, 60.0], vec![0.0]);
        let horizon = 6000.0; // just over one revolution

        let rows = coverage_metrics(a, 0.0, 0.0, 0.0, &sensor, cells.clone(), horizon, 1.0, 2);
        assert_eq!(rows.len(), 2, "one row per cell");

        let (equator, high_lat) = (rows[0], rows[1]);
        assert!(
            equator[0] >= 1.0,
            "the equatorial cell is never seen: {equator:?}"
        );
        assert!(
            equator[1] > 0.0 && equator[1] <= 1.0,
            "coverage fraction out of range"
        );
        assert!(
            equator[2] >= equator[3],
            "max gap is below the mean gap: {equator:?}"
        );
        assert_eq!(
            high_lat[0], 0.0,
            "an equatorial orbit cannot see 60°N: {high_lat:?}"
        );
        assert_eq!(
            high_lat[1], 0.0,
            "a cell with no pass has no coverage: {high_lat:?}"
        );

        // The same geometry through the single-cell window API: the satellite starts overhead, so
        // the first window opens at t = 0 and every window is non-empty and ordered.
        let windows = access_windows(a, 0.0, 0.0, 0.0, &sensor, cells[0], horizon, 1.0);
        assert!(!windows.is_empty(), "the overhead pass produced no window");
        assert_eq!(
            windows.len() as f64,
            equator[0],
            "the two APIs disagree on the pass count"
        );
        assert_eq!(
            windows[0].0, 0.0,
            "the satellite starts overhead, so the first rise is t = 0"
        );
        for w in &windows {
            assert!(w.1 > w.0, "a window that does not open: {w:?}");
            assert!(w.1 <= horizon, "a window past the horizon: {w:?}");
        }
    }

    /// The equatorial case above cannot see the orbit-plane rotation at all: with `inc = raan =
    /// phase = 0` every cross term in `sat_at` is multiplied by `sin i = 0` or `sin Ω = 0`, so the
    /// analytic ECI→ECEF propagation reduces to `a·(cos u, sin u, 0)` and a mutation of any of the
    /// six products is unobservable. Measured, not guessed: a run scoped to these two functions
    /// left 27 survivors, all of them in that closure and in the mean motion.
    ///
    /// So this one is inclined, with a non-zero node and phase, and the bridge's expanded closed
    /// form is checked against the orbit-plane basis it is the expansion of — the same geometry
    /// built a different way, which is what makes the comparison worth making. The elevation is
    /// then sampled directly, and every sample has to agree with the windows the engine reported.
    #[test]
    fn the_access_windows_agree_with_the_elevation_of_an_inclined_orbit() {
        let (a, inc, raan, phase) = (RE_EARTH + 550e3, 0.95, 0.6, 0.4);
        let horizon = 5000.0;
        let mask = 5f64.to_radians();
        let sensor = Sensor {
            kind: SensorKind::Elevation(ElevationMask::new(mask).expect("5° is a valid mask")),
        };

        // r_ECI(u) = a·(cos u · P̂ + sin u · Q̂), with P̂ the ascending node and Q̂ = ẑ_orbit × P̂ —
        // the perifocal basis written out, rather than the expanded product the bridge evaluates.
        // Then into ECEF by the Earth's rotation, which is what makes the ground track a track.
        let n = (MU_EARTH / (a * a * a)).sqrt();
        let sat_ecef = |t: f64| {
            let u = phase + n * t;
            let p = [raan.cos(), raan.sin(), 0.0];
            let q = [-raan.sin() * inc.cos(), raan.cos() * inc.cos(), inc.sin()];
            let eci = [
                a * (u.cos() * p[0] + u.sin() * q[0]),
                a * (u.cos() * p[1] + u.sin() * q[1]),
                a * (u.cos() * p[2] + u.sin() * q[2]),
            ];
            let th = EARTH_ROTATION_RATE * t;
            [
                eci[0] * th.cos() + eci[1] * th.sin(),
                -eci[0] * th.sin() + eci[1] * th.cos(),
                eci[2],
            ]
        };

        // The cell is the sub-satellite point at t = 0, so the pass is guaranteed and starts overhead.
        let r0 = sat_ecef(0.0);
        let norm0 = (r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2]).sqrt();
        let zenith = [r0[0] / norm0, r0[1] / norm0, r0[2] / norm0];
        let centre = [
            zenith[0] * RE_EARTH,
            zenith[1] * RE_EARTH,
            zenith[2] * RE_EARTH,
        ];

        let elevation_at = |t: f64| {
            let s = sat_ecef(t);
            let d = [s[0] - centre[0], s[1] - centre[1], s[2] - centre[2]];
            let range = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            // Clamped: straight overhead the quotient rounds a hair above 1 and `asin` returns NaN.
            ((d[0] * zenith[0] + d[1] * zenith[1] + d[2] * zenith[2]) / range)
                .clamp(-1.0, 1.0)
                .asin()
        };
        // 1e-6 rad, not 1e-9: `asin` has a square-root singularity at 1, so the 2e-16 of rounding in
        // the quotient reaches the angle as 2e-8 whatever the geometry does.
        assert!(
            (elevation_at(0.0) - FRAC_PI_2).abs() < 1e-6,
            "the cell is not the sub-satellite point at t = 0: {} °",
            elevation_at(0.0).to_degrees()
        );

        let windows = access_windows(a, inc, raan, phase, &sensor, (centre, zenith), horizon, 1.0);
        assert!(!windows.is_empty(), "the overhead pass produced no window");

        // Every sample: in a window ⟺ above the mask. Samples within the root-finder's tolerance of
        // a boundary are skipped, because there the two are allowed to disagree by construction.
        let (mut inside, mut outside) = (0, 0);
        let mut t = 0.0;
        while t <= horizon {
            let elevation = elevation_at(t);
            let in_window = windows.iter().any(|w| t >= w.0 && t <= w.1);
            let near_edge = windows
                .iter()
                .any(|w| (t - w.0).abs() < 2.0 || (t - w.1).abs() < 2.0);
            if !near_edge {
                assert_eq!(
                    in_window,
                    elevation >= mask,
                    "at t = {t} the engine says {in_window} but the elevation is {} °",
                    elevation.to_degrees()
                );
                if in_window {
                    inside += 1;
                } else {
                    outside += 1;
                }
            }
            t += 10.0;
        }
        assert!(
            inside > 10 && outside > 10,
            "the sampling did not straddle the mask: {inside} in, {outside} out"
        );

        // And the aggregate API counts the same passes over the same cell.
        let rows = coverage_metrics(
            a,
            inc,
            raan,
            phase,
            &sensor,
            vec![(centre, zenith)],
            horizon,
            1.0,
            2,
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0][0],
            windows.len() as f64,
            "the two APIs disagree on the pass count"
        );
        let visible: f64 = windows.iter().map(|w| w.1 - w.0).sum();
        assert!(
            (rows[0][1] - visible / horizon).abs() < 1e-6,
            "the coverage fraction is not the visible time over the horizon: {:?}",
            rows[0]
        );
    }

    /// The complex-list convention every spacesignal wrapper crosses the boundary with: two parallel
    /// `f64` lists, real first. `to_sig` and `from_sig` are inverses, and neither swaps the parts —
    /// which a symmetric fixture could not tell apart, so the parts are distinct and of opposite sign.
    #[test]
    fn the_complex_split_is_real_first_and_round_trips() {
        let (re, im) = (vec![1.0, -2.5, 0.0], vec![-3.0, 4.0, 7.5]);
        let z = to_sig(&re, &im);
        assert_eq!((z[0].re, z[0].im), (1.0, -3.0));
        assert_eq!((z[1].re, z[1].im), (-2.5, 4.0));
        assert_eq!(from_sig(&z), (re, im));
        // Ragged input truncates to the shorter list rather than panicking across the boundary.
        assert_eq!(to_sig(&[1.0, 2.0], &[3.0]).len(), 1);
    }

    /// `s₁·conj(s₂)`, on integers whose product is exact in binary: (1+2i)(3−4i) = 11+2i, and
    /// (−1+0i)·conj(0+1i) = +1i. A conjugate dropped, a factor swapped or a sum for a product all
    /// move the answer off those two, and the second pair is where the conjugate's sign shows.
    #[test]
    fn the_interferogram_is_the_conjugate_product_of_the_two_images() {
        let (re, im) = interferogram(
            vec![1.0, -1.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
            vec![4.0, 1.0],
        );
        assert_eq!((re, im), (vec![11.0, 0.0], vec![2.0, 1.0]));
    }

    /// The LFM chirp `exp(iπ·k·t²)` on `t = (i − (n−1)/2)/f_s`. Three separate properties: unit
    /// modulus everywhere (it is a phase-only waveform), the exact `1 + 0i` at the centre sample
    /// where `t = 0`, and the evenness in `t` that the squared time gives it. A wrong `t` origin
    /// breaks the last two, a wrong rate breaks the phase, and a magnitude that is not 1 breaks the
    /// first — no single fixture value to copy.
    #[test]
    fn the_lfm_chirp_is_a_unit_modulus_waveform_even_about_its_centre() {
        let (n, k, fs) = (9, 1e10, 1e6);
        let (re, im) = lfm_chirp(n, k, fs);
        assert_eq!(re.len(), n);
        for i in 0..n {
            assert!(
                (re[i].hypot(im[i]) - 1.0).abs() < 1e-15,
                "sample {i} is not on the unit circle"
            );
            assert!((re[i] - re[n - 1 - i]).abs() < 1e-15 && (im[i] - im[n - 1 - i]).abs() < 1e-15);
        }
        assert_eq!((re[4], im[4]), (1.0, 0.0));
        let t0 = -4.0 / fs;
        let (c, s) = (core::f64::consts::PI * k * t0 * t0).sin_cos();
        assert!((re[0] - s).abs() < 1e-12 && (im[0] - c).abs() < 1e-12);
    }

    /// Compressing a chirp against its own replica: the correlation peaks at zero lag with the
    /// signal's total energy, `n` for a unit-modulus waveform, and the peak is real. The output
    /// length is the linear-correlation length rounded up for the FFT, which is what tells a
    /// correlation from a convolution if the replica were not conjugated.
    #[test]
    fn the_matched_filter_compresses_a_chirp_to_its_own_energy_at_zero_lag() {
        let (n, k, fs) = (32, 3e10, 1e6);
        let (re, im) = lfm_chirp(n, k, fs);
        let (out_re, out_im) = matched_filter(re.clone(), im.clone(), re, im);
        assert_eq!(out_re.len(), (2 * n).next_power_of_two());
        assert!(
            (out_re[0] - n as f64).abs() < 1e-9,
            "zero-lag peak is {}",
            out_re[0]
        );
        assert!(
            out_im[0].abs() < 1e-9,
            "the autocorrelation peak must be real"
        );
        let elsewhere = (1..out_re.len())
            .map(|i| out_re[i].hypot(out_im[i]))
            .fold(0.0, f64::max);
        // Measured 3.37 against the peak's 32 — the ~9.5 compression the time-bandwidth product
        // of this chirp buys. The bound sits between, so a flat or shifted output fails it.
        assert!(
            elsewhere < 0.25 * n as f64,
            "the peak is not a peak: {elsewhere}"
        );
    }

    /// `T₃ = ⟨k·kᴴ⟩` over the Pauli vector `k = [(S_hh+S_vv), (S_hh−S_vv), 2·S_hv]/√2`, row-major and
    /// real-part-first. A pure single-bounce target — `S_hh = S_vv = 1`, `S_hv = 0` — has
    /// `k = [√2, 0, 0]`, so `T₃` is `diag(2, 0, 0)` exactly. That fixture pins the argument order:
    /// swapping `hv` for `vv` moves the energy off the first diagonal entry.
    #[test]
    fn the_coherency_matrix_is_the_outer_product_of_the_pauli_vector() {
        let (re, im) = coherency_matrix(vec![(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)]);
        // 2.0 to within the round-trip through 1/√2, which is not exact in binary.
        assert!((re[0] - 2.0).abs() < 1e-15, "T[0][0] is {}", re[0]);
        assert!(
            re[1..].iter().all(|x| *x == 0.0),
            "off-diagonal energy: {re:?}"
        );
        assert_eq!(im, vec![0.0; 9]);
        // Two samples average rather than accumulate: the same target twice is the same matrix.
        let (re2, _) = coherency_matrix(vec![
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        ]);
        assert!(
            (re2[0] - 2.0).abs() < 1e-15,
            "two identical samples average to {}",
            re2[0]
        );
    }

    /// The two ends of the Cloude–Pottier scale, which need no reference implementation because the
    /// entropy of an eigenvalue distribution is fixed at both: a rank-one `T₃` is one scattering
    /// mechanism, `H = 0`, and the identity is three equal ones, `H = 1`. Anisotropy is `0` in both —
    /// there is nothing to distinguish between the second and third eigenvalues either way — so the
    /// case that separates them is the third, `diag(4, 2, 0)`, where `A = (2−0)/(2+0) = 1`.
    #[test]
    fn the_cloude_pottier_entropy_runs_from_a_single_mechanism_to_three_equal_ones() {
        let diag = |a: f64, b: f64, c: f64| {
            let mut t = vec![0.0; 9];
            (t[0], t[4], t[8]) = (a, b, c);
            cloude_pottier(t, vec![0.0; 9])
        };
        let (h, a, alpha) = diag(1.0, 0.0, 0.0);
        assert!(
            h.abs() < 1e-12 && a.abs() < 1e-12 && alpha.abs() < 1e-12,
            "rank one: {h} {a} {alpha}"
        );
        let (h, a, _) = diag(1.0, 1.0, 1.0);
        assert!(
            (h - 1.0).abs() < 1e-12 && a.abs() < 1e-12,
            "isotropic: {h} {a}"
        );
        let (_, a, _) = diag(4.0, 2.0, 0.0);
        assert!((a - 1.0).abs() < 1e-12, "anisotropy of diag(4,2,0) is {a}");
    }

    /// Harris on a bright square: four corners, at the pixels the square's own extent puts them.
    /// The fixture is asymmetric on purpose — 10..25 across and 12..27 down — because a transposed
    /// binding would return `(13, 11)` for `(11, 13)` and a symmetric square could not tell them
    /// apart. The response is the third field and is strictly positive at a corner, which is what
    /// separates it from the `(x, y)` pair.
    #[test]
    fn the_harris_binding_finds_the_four_corners_of_a_square_in_pixel_order() {
        let (h, w) = (40usize, 40usize);
        let mut image = vec![0.0; h * w];
        for y in 12..28 {
            for x in 10..26 {
                image[y * w + x] = 1.0;
            }
        }
        let corners = harris_corners(image, h, w, 0.04, 2, 0.01, 3, 50);
        let mut found: Vec<(i64, i64)> = corners
            .iter()
            .map(|c| (c.0.round() as i64, c.1.round() as i64))
            .collect();
        found.sort_unstable();
        assert_eq!(found, [(11, 13), (11, 26), (24, 13), (24, 26)]);
        assert!(corners.iter().all(|c| c.2 > 0.0), "responses: {corners:?}");
    }

    /// SL1MMER against a measurement it built itself: one scatterer at a grid node, so
    /// `y = a(s₀)` with `a_i = exp(−i·4π·b_i·s₀/(λ·R))`, and the sparse solve has to put it back
    /// where it came from. The elevation is exact because it is a grid node; the reflectivity is
    /// only bounded from below, because an L1 penalty shrinks the amplitude it recovers — asserting
    /// it exactly would be asserting the value of `lambda`.
    #[test]
    fn the_sl1mmer_binding_recovers_the_scatterer_it_was_given() {
        let (wavelength, range, truth) = (0.056, 700e3, 40.0);
        let baselines = vec![-100.0, -66.0, -33.0, 0.0, 33.0, 66.0, 100.0];
        let (y_re, y_im): (Vec<f64>, Vec<f64>) = baselines
            .iter()
            .map(|b| {
                let phi = -4.0 * core::f64::consts::PI * b * truth / (wavelength * range);
                (phi.cos(), phi.sin())
            })
            .unzip();
        let elevations: Vec<f64> = (-40..=40).map(|s| f64::from(s) * 5.0).collect();
        let found = sl1mmer(
            y_re, y_im, baselines, wavelength, range, elevations, 0.4, 4_000, 0.5,
        );
        assert_eq!(found.len(), 1, "one scatterer in, {found:?} out");
        assert_eq!(
            found[0].0, truth,
            "the elevation is a grid node and must come back exact"
        );
        assert!(
            found[0].1.hypot(found[0].2) > 0.5,
            "the reflectivity shrank to nothing: {:?}",
            found[0]
        );
    }

    /// Chirp scaling multiplies by `exp(iπ·k·α·(m − τ_ref)²/f_s²)`, which is a phase and nothing
    /// else. Three properties fix it without a reference array: every magnitude survives, the sample
    /// at `τ_ref` is returned untouched because its phase is exactly zero there, and the multiplier
    /// is even about `τ_ref` because the time enters squared. A wrong origin breaks the last two and
    /// a multiplier that is not on the unit circle breaks the first.
    #[test]
    fn chirp_scaling_is_a_phase_only_multiply_centred_on_its_reference() {
        let (n, fs) = (64, 1.0e8);
        let (cr, ci) = lfm_chirp(n, 5.0e13, fs);
        let tau_ref = 30.0;
        let (sr, si) = apply_chirp_scaling(cr.clone(), ci.clone(), 5.0e13, 0.05, tau_ref, fs);
        for m in 0..n {
            assert!(
                (sr[m].hypot(si[m]) - cr[m].hypot(ci[m])).abs() < 1e-12,
                "sample {m} changed magnitude"
            );
        }
        let r = tau_ref as usize;
        assert!((sr[r] - cr[r]).abs() < 1e-15 && (si[r] - ci[r]).abs() < 1e-15);
        // The multiplier at τ_ref ± d is the same, so the ratio out/in is equal on both sides.
        let ratio = |m: usize| {
            let (a, b) = (cr[m], ci[m]);
            let d = a * a + b * b;
            ((sr[m] * a + si[m] * b) / d, (si[m] * a - sr[m] * b) / d)
        };
        for d in 1..20 {
            let (lo, hi) = (ratio(r - d), ratio(r + d));
            assert!(
                (lo.0 - hi.0).abs() < 1e-12 && (lo.1 - hi.1).abs() < 1e-12,
                "the multiplier is not even about the reference at offset {d}"
            );
        }
    }

    /// The full chirp-scaling pipeline on a point target that migrates across range cells over the
    /// aperture: it has to come back as one sharp peak in the reference-range column. Sharpness is
    /// asserted as peak-over-mean, which is scale-invariant and so says nothing about the gain, and
    /// as the count of cells within 6 dB of the peak — a smeared answer fails both while a constant
    /// output fails the first outright.
    #[test]
    fn the_chirp_scaling_pipeline_focuses_a_migrating_point_target() {
        let (na, nr) = (128usize, 64usize);
        let (prf, vel, lam, r0, fs, kr) = (500.0, 7500.0, 0.3, 500e3, 150e6, 2.0e14);
        let (pulse, c, dtr) = (0.5e-6, 299_792_458.0, 1.0 / 150e6);
        let (mut re, mut im) = (vec![0.0; na * nr], vec![0.0; na * nr]);
        for a in 0..na {
            let eta = (a as f64 - na as f64 / 2.0) / prf;
            let rng = (r0 * r0 + vel * vel * eta * eta).sqrt();
            let mig = 2.0 * (rng - r0) / c;
            let azph = -4.0 * core::f64::consts::PI * rng / lam;
            for rr in 0..nr {
                let tau = (rr as f64 - nr as f64 / 2.0) * dtr - mig;
                if tau.abs() < pulse / 2.0 {
                    let (s, cph) = (azph + core::f64::consts::PI * kr * tau * tau).sin_cos();
                    (re[a * nr + rr], im[a * nr + rr]) = (cph, s);
                }
            }
        }
        let (fr, fi) = chirp_scaling_focus(re, im, na, nr, prf, vel, lam, r0, fs, kr);
        assert_eq!((fr.len(), fi.len()), (na * nr, na * nr));
        let mag: Vec<f64> = (0..na * nr).map(|i| fr[i].hypot(fi[i])).collect();
        let peak = mag.iter().copied().fold(0.0, f64::max);
        let mean = mag.iter().sum::<f64>() / mag.len() as f64;
        assert!(peak > 100.0 * mean, "peak {peak} against mean {mean}");
        assert!(
            mag.iter().filter(|m| **m > 0.5 * peak).count() < 8,
            "the target is smeared"
        );
        let at = mag.iter().position(|m| *m == peak).expect("a finite peak");
        assert_eq!(
            at % nr,
            nr / 2,
            "the peak is not in the reference-range column"
        );
    }

    /// The SL1MMER spectrum on the same self-built measurement as the full solve: one value per grid
    /// node, peaking at the node the scatterer was placed on. Length and peak position together are
    /// what a constant or empty return cannot fake.
    #[test]
    fn the_sl1mmer_spectrum_peaks_at_the_scatterer_it_was_given() {
        let (wavelength, range, truth) = (0.056, 700e3, 40.0);
        let baselines = vec![-100.0, -66.0, -33.0, 0.0, 33.0, 66.0, 100.0];
        let (y_re, y_im): (Vec<f64>, Vec<f64>) = baselines
            .iter()
            .map(|b| {
                let phi = -4.0 * core::f64::consts::PI * b * truth / (wavelength * range);
                (phi.cos(), phi.sin())
            })
            .unzip();
        let elevations: Vec<f64> = (-40..=40).map(|s| f64::from(s) * 5.0).collect();
        let spectrum = sl1mmer_spectrum(
            y_re,
            y_im,
            baselines,
            wavelength,
            range,
            elevations.clone(),
            0.4,
            4_000,
        );
        assert_eq!(spectrum.len(), elevations.len());
        let at = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("a non-empty spectrum")
            .0;
        assert_eq!(
            elevations[at], truth,
            "the spectrum peaks at {}",
            elevations[at]
        );
    }

    /// The NRLMSISE-00 wrapper's job is to fill the model's input record and convert its units, so
    /// the oracle is the model itself called with a record built here: the local solar time is
    /// `sec/3600 + lon/15` — hours of UT plus hours of longitude — and the output is `d[5]` in g/cm³
    /// scaled to kg/m³. A longitude 15° east of the reference at an hour earlier is the same local
    /// time, which is the second, unit-free way the same wiring shows.
    #[test]
    fn the_density_wrapper_fills_the_model_record_and_converts_to_si() {
        let (doy, sec, alt, lat, lon) = (172, 29_000.0, 400.0, 60.0, -70.0);
        let (f107a, f107, ap) = (150.0, 150.0, 4.0);
        let reference = gtd7d(
            &Nrlmsise00Input {
                year: 0,
                doy,
                sec,
                alt,
                g_lat: lat,
                g_long: lon,
                lst: sec / 3600.0 + lon / 15.0,
                f107a,
                f107,
                ap,
                ap_array: None,
            },
            &Nrlmsise00Flags::default(),
        )
        .d[5];
        let got = nrlmsise00_density(doy, sec, alt, lat, lon, f107, f107a, ap);
        assert!(
            (got - reference * 1000.0).abs() < 1e-24,
            "{got} against {reference} g/cm³"
        );
        // Thermospheric density at 400 km is ~1e-12 kg/m³; in g/cm³ it would be ~1e-15. The bound
        // spans three decades either way, so it fixes the conversion without pinning the model.
        assert!(
            (1e-14..1e-10).contains(&got),
            "not a kg/m³ thermospheric density: {got}"
        );
        // An hour earlier, fifteen degrees east: the same local solar time, so the diurnal terms
        // land in the same place and only the explicit `sec` moves the answer at all.
        let shifted = nrlmsise00_density(doy, sec - 3600.0, alt, lat, lon + 15.0, f107, f107a, ap);
        assert!(
            (shifted / got - 1.0).abs() < 0.05,
            "same local time gave {shifted} against {got}"
        );
    }

    /// The bridge's SDTM-1 wrapper does one thing the model does not: it assembles the evaluation
    /// point out of the coordinates a caller already has. Each of its four conversions is asserted as
    /// a **value**, because the metamorphic form — two calls a half-day and 180 degrees apart, which
    /// name the same point and so must agree — is blind to the longitude term's scale *and* its sign
    /// at once: 360 degrees becomes 5 400 hours under `longitude · 15`, and 24 divides 5 400. A
    /// symmetry cannot pin the constant the symmetry is built out of.
    #[test]
    fn the_sdtm1_wrapper_hands_the_field_the_point_the_fit_was_written_over() {
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
        // the record by name and the binding builds it from its argument order.
        let reference = SDTM1_FIELD.density(
            &p,
            &SdtmDrivers {
                f107: 120.0,
                f107a: 160.0,
                storm: 30.0,
            },
        );
        let got = sdtm1_density(172.0, 54_000.0, 400.0, 45.0, 30.0, 120.0, 160.0, 30.0);
        assert_eq!(got, reference, "{got} against {reference}");
        // kg/m³ at 400 km and moderate flux, so the km → m conversion is pinned by the value too.
        assert!((1e-13..1e-11).contains(&got), "{got}");

        // And the profile is a thinning one, which a single point cannot say.
        let at =
            |alt_km: f64| sdtm1_density(172.0, 54_000.0, alt_km, 45.0, 30.0, 120.0, 160.0, 30.0);
        assert!(at(200.0) > got && got > at(600.0));
    }

    /// `T∞` is the quantity the fit parameterises; the density is a Bates profile above it. The range
    /// is the shipped field's own envelope over the six months it was fitted on, and the storm
    /// direction is the sign its coefficients carry.
    #[test]
    fn the_sdtm1_exospheric_temperature_rises_with_the_storm_driver() {
        let at = |storm: f64| {
            sdtm1_exospheric_temperature(172.0, 29_000.0, 400.0, 0.0, 0.0, 150.0, 150.0, storm)
        };
        let quiet = at(0.0);
        assert!((586.0..1528.0).contains(&quiet), "{quiet} K");
        assert!(at(100.0) > quiet, "{} against {quiet}", at(100.0));
    }

    /// The shipped storm state cools, so a constant history answers its steady state
    /// `H/(1 + H/H_s)` from the first step, at any cadence — the field's own fixed point, which is
    /// what makes it worth asserting in a wrapper.
    #[test]
    fn the_sdtm1_storm_state_holds_a_constant_history_at_its_steady_state() {
        let spacelab_propagation::sdtm::StormResponse::Cooling(cooling) = SDTM1_FIELD.storm()
        else {
            panic!("the shipped field's storm state is the cooling one");
        };
        let steady = cooling.steady_state(5.0);
        assert!(steady < 5.0);
        for cadence in [10_800.0, 3_600.0] {
            let got = sdtm1_storm_response(vec![5.0; 16], cadence);
            assert!(
                (got - steady).abs() < 1e-12 * steady,
                "{got} at {cadence} s"
            );
        }
        // No history is no response, which is the answer an empty Python list gets.
        assert_eq!(sdtm1_storm_response(Vec::new(), 10_800.0), 0.0);
        // A surge over a quiet background lands between the background and the peak.
        let mut surge = vec![5.0; 16];
        surge[12..].fill(300.0);
        let response = sdtm1_storm_response(surge, 10_800.0);
        assert!((steady..300.0).contains(&response), "{response}");
    }

    /// The interval is log-normal about the density, so `lower · upper = density²` exactly whatever
    /// the regime and whatever `z`. That is the one property common to all six fitted sigmas, so it
    /// checks the wrapper without pinning the table.
    #[test]
    fn the_sdtm1_interval_is_log_normal_about_the_density_it_brackets() {
        let density = 4.118e-12;
        let (lower, upper) =
            sdtm1_density_interval(density, 150.0, 150.0, 0.0, 1.644_853_626_951_472_7);
        assert!(
            lower < density && density < upper,
            "{lower} {density} {upper}"
        );
        assert!(
            (lower * upper / (density * density) - 1.0).abs() < 1e-12,
            "{lower} x {upper} is not {density} squared"
        );
        // `z = 0` is the only multiplier for which the two ends coincide, so it pins the multiplier's
        // place in the exponent.
        assert_eq!(
            sdtm1_density_interval(density, 150.0, 150.0, 0.0, 0.0),
            (density, density)
        );
        // The quiet low-flux cell is wider than the quiet high-flux one, so the flux edge is read.
        let (quiet_lo, quiet_hi) =
            sdtm1_density_interval(density, 80.0, 80.0, 0.0, 1.644_853_626_951_472_7);
        assert!(
            quiet_hi / quiet_lo > upper / lower,
            "{quiet_lo}..{quiet_hi} against {lower}..{upper}"
        );
        // Moving the storm bin moves the interval; the measured direction is not asserted, only that
        // the bin is read at all.
        assert_ne!(
            sdtm1_density_interval(density, 150.0, 150.0, 100.0, 1.644_853_626_951_472_7),
            (lower, upper)
        );
    }

    /// The envelope in the bridge's units, kilometres and solar flux units, in the documented order.
    #[test]
    fn the_sdtm1_envelope_is_the_fitted_one_in_kilometres() {
        let ((altitude_lo, altitude_hi), (f107a_lo, f107a_hi)) = sdtm1_envelope();
        assert_eq!(
            [altitude_lo, altitude_hi, f107a_lo, f107a_hi],
            [
                *SDTM1_ENVELOPE.altitude.start() / 1000.0,
                *SDTM1_ENVELOPE.altitude.end() / 1000.0,
                *SDTM1_ENVELOPE.f107a.start(),
                *SDTM1_ENVELOPE.f107a.end(),
            ]
        );
        assert!(
            (200.0..400.0).contains(&altitude_lo) && (400.0..600.0).contains(&altitude_hi),
            "{altitude_lo}..{altitude_hi} km"
        );
    }

    /// The geodetic triple both drag atmospheres read, against the same rotation applied through the
    /// typed frame API instead of by multiplying the three rows out by hand. A garbled row moves the
    /// altitude by hundreds of kilometres, and a dropped `/ 1000` by a factor of a thousand.
    #[test]
    fn the_drag_geodetic_point_is_the_typed_frame_transforms() {
        let (epoch_tt, epoch_ut1) = (631_152_069_184_000_000_i128, 631_152_000_000_000_000_i128);
        let cio = || cio_fast(epoch_tt, epoch_ut1, 3_600.0, 1e-6, 1e-6, 0.0, 0.0);
        let (typed, geodetic) = (cio(), geodetic_over(cio()));
        let r = [4.1e6, 3.3e6, 3.8e6];
        for t in [0.0, 1_234.5, 3_600.0] {
            let g =
                earth_itrf_to_geodetic(typed.transform(t).position(Position::<Gcrf>::new(vec3(r))))
                    .expect("the point is above the ellipsoid, so it has a geodetic position");
            let (altitude_km, latitude_deg, longitude_deg) = geodetic(t, r);
            assert!(
                (altitude_km - g.height.metres() / 1000.0).abs() < 1e-9
                    && (latitude_deg - g.latitude.degrees()).abs() < 1e-12
                    && (longitude_deg - g.longitude.degrees()).abs() < 1e-12,
                "t = {t}: {:?} against {g:?}",
                (altitude_km, latitude_deg, longitude_deg)
            );
        }
    }

    /// One `D19.12` RINEX field: sign/space, one leading digit, 12 fractional, `D`, signed 2-digit
    /// exponent. Same shape as the fixture builder in `spacelab-gnss`, so the columns the parser
    /// slices are the columns the spec puts them in.
    fn d19(x: f64) -> String {
        let s = format!("{x:.12E}");
        let (mantissa, exp) = s.split_once('E').expect("scientific notation has an E");
        let exp: i32 = exp.parse().expect("the exponent is an integer");
        let sign = if x < 0.0 { '-' } else { ' ' };
        let esign = if exp < 0 { '-' } else { '+' };
        format!(
            "{sign}{}D{esign}{:02}",
            mantissa.trim_start_matches('-'),
            exp.abs()
        )
    }

    /// One broadcast-orbit line. The indent is **four** spaces -- `4X` in the format's own notation
    /// -- and that is worth a comment because it used to be three here, in the C ABI, in the Python
    /// bridge and in the reader itself: four fixtures agreeing with the parser about the columns
    /// instead of with the format, so none of them could disagree with it about a value.
    fn orbit_line(a: f64, b: f64, c: f64, d: f64) -> String {
        format!("    {}{}{}{}", d19(a), d19(b), d19(c), d19(d))
    }

    /// Eight lines of one Keplerian record, tagged `sys` at `prn`, at a medium-Earth-orbit radius.
    fn keplerian_record(sys: char, prn: u8) -> String {
        format!(
            "{sys}{prn:02} 2024 03 05 12 34 56{c0}{c1}{c2}\n{o1}\n{o2}\n{o3}\n{o4}\n{o5}\n{o6}\n{o7}",
            c0 = d19(1.5e-4),
            c1 = d19(3.4e-12),
            c2 = d19(0.0),
            o1 = orbit_line(1.0, 35.0, 4.5e-9, 0.321),
            o2 = orbit_line(1.2e-6, 0.006_2, 8.3e-6, 5153.65),
            o3 = orbit_line(216_000.0, -1.1e-7, -1.234, 9.0e-8),
            o4 = orbit_line(0.959_9, 210.0, 0.567, -8.1e-9),
            o5 = orbit_line(-2.1e-10, 1.0, 2300.0, 0.0),
            o6 = orbit_line(0.0, 0.0, -1.0e-8, 40.0),
            o7 = orbit_line(216_000.0, 4.0, 0.0, 0.0),
        )
    }

    fn with_header(body: &str) -> String {
        format!(
            "     3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE\n\
             {}END OF HEADER\n{body}\n",
            " ".repeat(60),
        )
    }

    /// The bridge propagates *every* Keplerian constellation in a mixed file, in file order, and
    /// tags each row with the RINEX letter -- without which PRN 7 could be any of five satellites.
    /// The oracle is the parser and the orbit model called directly, so the assertion is exactly on
    /// the marshalling: the same `tk`, the same records, bit-identical states.
    #[test]
    fn the_rinex_bridge_keeps_every_constellation_and_labels_its_prn() {
        let text = with_header(&format!(
            "{}\n{}\n{}\n{}",
            keplerian_record('G', 7),
            keplerian_record('E', 11),
            keplerian_record('C', 21),
            glonass_record(),
        ));
        let tk = 300.0;

        let got = rinex_gnss_positions(&text, tk);
        let reference = parse_nav(&text);
        assert_eq!(reference.len(), 3, "three Keplerian records were written");
        assert_eq!(got.len(), reference.len(), "a GNSS record was dropped");
        assert_eq!(
            got.iter().map(|r| (r.0, r.1)).collect::<Vec<_>>(),
            vec![('G', 7), ('E', 11), ('C', 21)],
            "the constellation letters or the PRNs are wrong"
        );
        for ((sys, prn, p, v), eph) in got.iter().zip(&reference) {
            let (rp, rv) = eph.position_velocity_ecef(tk);
            assert_eq!(
                (*p, *v),
                (rp, rv),
                "{sys}{prn} is not the model's own state"
            );
            let radius = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!(
                (26.0e6..27.2e6).contains(&radius),
                "{sys}{prn} sits at {radius} m, not in medium Earth orbit"
            );
        }

        // `tk` is seconds from each record's reference epoch, so it has to move the satellite.
        let later = rinex_gnss_positions(&text, tk + 600.0);
        assert!(
            (later[0].2[0] - got[0].2[0]).abs() > 1.0e3,
            "tk is not reaching the orbit model"
        );
    }

    /// The four lines of one GLONASS record at slot 5, PZ-90 in km.
    fn glonass_record() -> String {
        format!(
            "R05 2024 01 09 03 15 42{c0}{c1}{c2}\n{l1}\n{l2}\n{l3}",
            c0 = d19(-1.5e-4),
            c1 = d19(0.0),
            c2 = d19(11_742.0),
            l1 = orbit_line(25_510.0, 0.1, 1.0e-9, 0.0),
            l2 = orbit_line(-1_200.0, 2.9, -2.0e-9, -1.0),
            l3 = orbit_line(800.0, 1.3, 3.0e-9, 0.0),
        )
    }

    /// The GLONASS half of the same wiring: the slot, the PZ-90 state from the crate's own
    /// integration, and the Keplerian records in the same file stepped over rather than parsed.
    #[test]
    fn the_glonass_bridge_carries_the_slot_and_the_pz90_state() {
        let text = with_header(&format!(
            "{}\n{}",
            keplerian_record('G', 7),
            glonass_record()
        ));
        let dt = 900.0;

        let got = rinex_glonass_positions(&text, dt);
        let reference = parse_glonass_nav(&text);
        assert_eq!(got.len(), 1, "one GLONASS record, and no Keplerian one");
        assert_eq!(reference.len(), 1);
        let (rp, rv) = reference[0].position_velocity_pz90(dt);
        assert_eq!(got[0], (reference[0].slot, rp, rv));
        assert_eq!(got[0].0, 5, "the slot number was lost");

        let radius = (rp[0] * rp[0] + rp[1] * rp[1] + rp[2] * rp[2]).sqrt();
        assert!(
            (24.5e6..26.5e6).contains(&radius),
            "{radius} m is not the GLONASS shell -- the km->m scaling is wrong"
        );
        // 900 s at the broadcast |v| ~ 3.2 km/s is a ~2900 km arc; a `dt` that never reached the
        // integrator would leave the satellite on its epoch state.
        let at_epoch = rinex_glonass_positions(&text, 0.0);
        let moved = (0..3)
            .map(|k| (at_epoch[0].1[k] - rp[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(moved > 1.0e6, "dt moved the satellite only {moved} m");
    }

    /// The Street-of-Coverage binding has to emit exactly the constellation its own geometry
    /// function reports: `planes x per_plane` satellites, every one of them on the circular orbit
    /// the inputs describe, with the plane RAANs stepping by the reported `raan_step`. An
    /// infeasible geometry is the documented empty list, not a partial one.
    #[test]
    fn the_street_of_coverage_binding_emits_the_geometry_it_reports() {
        let (inc, alt, min_el) = (1.6, 800e3, 0.17);
        let (re, mu) = (6_378_137.0, 3.986_004_418e14);
        let (planes, per_plane, _, _, _, _, raan_step, _, _) =
            street_of_coverage_geometry(inc, alt, min_el, re, mu, None)
                .expect("a near-polar 800 km street tiles");
        let sats = street_of_coverage(inc, alt, min_el, re, mu, None);

        assert_eq!(
            sats.len() as u32,
            planes * per_plane,
            "wrong satellite count"
        );
        assert!(planes >= 2 && per_plane >= 2, "{planes} x {per_plane}");
        for (r, v) in &sats {
            let radius = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            assert!(
                (radius - (re + alt)).abs() < 1e-6,
                "{radius} m is not the circular radius {}",
                re + alt
            );
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!(
                (speed - (mu / radius).sqrt()).abs() < 1e-6,
                "{speed} m/s is not circular"
            );
        }
        // Plane-major: the first satellite of plane 1 is one `raan_step` round from plane 0's.
        let raan_of = |r: [f64; 3], v: [f64; 3]| {
            let h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ];
            h[0].atan2(-h[1])
        };
        let step = raan_of(sats[per_plane as usize].0, sats[per_plane as usize].1)
            - raan_of(sats[0].0, sats[0].1);
        assert!(
            (step - raan_step).abs() < 1e-9,
            "planes step by {step} rad, the geometry says {raan_step}"
        );

        // Both documented infeasibilities are the empty list, not a partial constellation: a
        // `per_plane` under the `⌈π/θ⌉` floor leaves gaps along the street, and an inclination this
        // far from polar cannot space the planes at all.
        assert!(street_of_coverage(inc, alt, min_el, re, mu, Some(2)).is_empty());
        assert!(street_of_coverage(0.05, alt, min_el, re, mu, None).is_empty());
    }

    /// The Flower binding phases `satellites` spacecraft along one `(petals, days)` repeating orbit,
    /// so every one of them shares the semi-major axis, the eccentricity and the inclination, and
    /// the axis is the one the repeat condition fixes: `Np` revolutions in `Nd` sidereal days.
    #[test]
    fn the_flower_binding_phases_one_repeating_orbit() {
        let (petals, days, satellites) = (5u32, 2u32, 8u32);
        let (ecc, inc, argp, mu) = (0.2, 0.5, 1.1, 3.986_004_418e14);
        let frame_rate = 7.292_115_146_5e-5;
        let sats = flower_constellation(
            petals, days, satellites, ecc, inc, argp, 0.0, 0.0, frame_rate, mu,
        );
        assert_eq!(sats.len(), satellites as usize);

        let period = f64::from(days) * (core::f64::consts::TAU / frame_rate) / f64::from(petals);
        let a_repeat = (mu * (period / core::f64::consts::TAU).powi(2)).cbrt();
        for (i, (r, v)) in sats.iter().enumerate() {
            let el = KeplerianElements::from_state(vec3(*r), vec3(*v), mu);
            assert!(
                (el.semi_major_axis / a_repeat - 1.0).abs() < 1e-9,
                "satellite {i} is on a {} m axis, the repeat needs {a_repeat} m",
                el.semi_major_axis
            );
            assert!((el.eccentricity - ecc).abs() < 1e-9, "satellite {i} shape");
            assert!(
                (el.inclination.radians() - inc).abs() < 1e-9,
                "satellite {i} inclination"
            );
        }
        // Time-phased, not stacked: no two satellites share a position.
        assert!(
            (sats[0].0[0] - sats[1].0[0]).abs() + (sats[0].0[1] - sats[1].0[1]).abs() > 1e3,
            "the first two satellites are on top of each other"
        );
        assert!(
            flower_constellation(petals, days, 0, ecc, inc, argp, 0.0, 0.0, frame_rate, mu)
                .is_empty()
        );
    }

    /// Draim's constellation is four satellites, and the binding has to put all four on the conic
    /// its own geometry function reports for the same period and eccentricity.
    #[test]
    fn the_draim_binding_is_four_satellites_on_the_conic_it_reports() {
        let (period, ecc, inc, mu) = (172_800.0, 0.28, 0.575_958_653_158_128_8, 3.986_004_418e14);
        let (a, rp, ra) = draim_geometry(period, ecc, inc, mu);
        let sats = draim_constellation(period, ecc, inc, 0.0, mu);
        assert_eq!(sats.len(), 4, "Draim is a four-satellite constellation");

        let mut raans = Vec::new();
        for (i, (r, v)) in sats.iter().enumerate() {
            let el = KeplerianElements::from_state(vec3(*r), vec3(*v), mu);
            assert!(
                (el.semi_major_axis / a - 1.0).abs() < 1e-9,
                "satellite {i} axis {} against {a}",
                el.semi_major_axis
            );
            assert!((el.eccentricity - ecc).abs() < 1e-9);
            assert!((el.inclination.radians() - inc).abs() < 1e-9);
            let radius = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            assert!(
                (rp - 1e-3..=ra + 1e-3).contains(&radius),
                "satellite {i} at {radius} m is off the conic [{rp}, {ra}]"
            );
            raans.push(el.raan.radians());
        }
        // Four distinct orbit planes -- a tetrahedral constellation, not four copies of one plane.
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert!(
                    (raans[i] - raans[j]).abs() > 1e-3,
                    "satellites {i} and {j} share a plane"
                );
            }
        }
    }

    /// The five tropospheric mapping bindings are pass-throughs, so the oracle is the crate function
    /// under the same arguments, bit for bit. The elevation ladder is what makes that non-vacuous:
    /// a mapping factor is 1 along the zenith and grows towards the horizon, so the pair is neither
    /// `(0, 0)` nor `(1, 1)` at the elevation the equality is taken at, and the wet factor leads the
    /// hydrostatic one -- water vapour sits in a thinner layer, so its mapping runs closer to the
    /// flat-atmosphere `1/sin ε` limit. Measured at 10 deg: 5.5549 against 5.6581 for Niell.
    #[test]
    fn the_tropospheric_mapping_bindings_are_their_crate_functions() {
        let (lat, lon, height, doy) = (core::f64::consts::FRAC_PI_4, 0.2618, 100.0, 90.0);
        let (a_h, a_w) = (1.23e-3, 5.6e-4);
        let el = 0.174_5; // 10 deg -- well off the zenith, where every factor is ~1

        type MappingCase = ((f64, f64), (f64, f64), &'static str);
        let cases: [MappingCase; _] = [
            (
                niell_mapping(lat, height, doy, el),
                gnss_niell_mapping(lat, height, doy, el),
                "niell",
            ),
            (
                gmf_mapping(lat, lon, height, doy, el),
                gnss_gmf_mapping(lat, lon, height, doy, el),
                "gmf",
            ),
            (
                vmf1_mapping(a_h, a_w, lat, height, doy, el),
                gnss_vmf1_mapping(a_h, a_w, lat, height, doy, el),
                "vmf1",
            ),
            (
                vmf3_mapping(a_h, a_w, lat, lon, height, doy, el),
                gnss_vmf3_mapping(a_h, a_w, lat, lon, height, doy, el),
                "vmf3",
            ),
        ];
        for ((got_h, got_w), (want_h, want_w), name) in cases {
            assert_eq!((got_h, got_w), (want_h, want_w), "{name} is not wired");
            assert!(
                (4.0..8.0).contains(&got_h) && (4.0..8.0).contains(&got_w),
                "{name} at 10 deg gave ({got_h}, {got_w}), not a ~5.6 slant factor"
            );
            assert!(
                got_w > got_h,
                "{name} put the hydrostatic factor above the wet one"
            );
        }

        // Zenith is the definition of 1, and the factors rise monotonically as the ray drops.
        let zenith = core::f64::consts::FRAC_PI_2;
        let (zh, zw) = niell_mapping(lat, height, doy, zenith);
        assert!(
            (zh - 1.0).abs() < 1e-9 && (zw - 1.0).abs() < 1e-9,
            "{zh}, {zw}"
        );
        let mut previous = 0.0;
        for step in 1..=8 {
            let (h, _) = niell_mapping(lat, height, doy, zenith / f64::from(step));
            assert!(h > previous, "the mapping fell at 1/{step} of the zenith");
            previous = h;
        }
    }

    /// The ionospheric bindings: NeQuick-G reproduces its standard's published test vector (Galileo
    /// OS-SIS-ICD), which is an oracle outside this repository; the rest are pass-throughs pinned
    /// against the crate call plus the `1/f²` and `1/f³` scalings that define them.
    #[test]
    fn the_ionospheric_bindings_reproduce_their_published_vectors_and_scalings() {
        // Galileo ICD Annex E, first case: 20.40 TECU printed, 20.40224 from the JRC reference.
        let stec = nequick_galileo_stec(
            [236.831_641, -0.393_628_78, 0.004_028_266_13],
            4,
            0.0,
            82.49_f64.to_radians(),
            297.66_f64.to_radians(),
            78.11,
            54.29_f64.to_radians(),
            8.23_f64.to_radians(),
            20_281_546.18,
        );
        assert!((stec - 20.40224).abs() < 1e-5, "NeQuick-G gave {stec} TECU");

        // First order is 40.3·TEC/f²: doubling the carrier quarters the delay.
        let (l1, l2) = (1_575.42e6, 1_227.60e6);
        let delay = ionospheric_path_delay(stec, l1);
        assert_eq!(delay, iono_path_delay(stec, l1));
        assert!(
            (delay - 40.3 * stec * 1e16 / (l1 * l1)).abs() < 1e-9,
            "{delay} m is not 40.3·TEC/f²"
        );
        assert!(
            (ionospheric_path_delay(stec, 2.0 * l1) / delay - 0.25).abs() < 1e-12,
            "the delay does not scale as 1/f²"
        );

        // Second order is the signed 1/f³ term, so L2 sees more of it than L1 by (f1/f2)³.
        let (lat, lon, el, az) = (0.6, -1.2, 0.5, 2.0);
        let hoi2 = second_order_ionospheric_delay(lat, lon, el, az, stec, l1, 350_000.0);
        assert_eq!(
            hoi2,
            gnss_hoi2(lat, lon, el, az, 350_000.0, stec * 1e16, l1)
        );
        assert!(hoi2.abs() > 1e-4, "the second-order term vanished: {hoi2}");
        let hoi2_l2 = second_order_ionospheric_delay(lat, lon, el, az, stec, l2, 350_000.0);
        assert!(
            (hoi2_l2 / hoi2 - (l1 / l2).powi(3)).abs() < 1e-9,
            "the second-order term does not scale as 1/f³"
        );

        // Klobuchar is a positive slant delay that grows as the satellite drops.
        let (alpha, beta) = ([1e-8, 0.0, 0.0, 0.0], [9e4, 0.0, 0.0, 0.0]);
        let high = klobuchar_iono_delay(alpha, beta, 0.5, 0.3, 1.2, 1.0, 50_400.0);
        let low = klobuchar_iono_delay(alpha, beta, 0.5, 0.3, 0.3, 1.0, 50_400.0);
        assert_eq!(
            high,
            gnss_klobuchar(alpha, beta, 0.5, 0.3, 1.2, 1.0, 50_400.0)
        );
        assert!(
            high > 0.0 && low > high,
            "{low} m at 17 deg against {high} m"
        );
    }

    /// The Saastamoinen zenith delay and the two GLONASS FDMA carriers: thin wrappers whose value is
    /// fixed by the formula in their own doc comment, so each is checked against that formula rather
    /// than against a remembered number.
    #[test]
    fn the_saastamoinen_and_glonass_bindings_match_their_documented_formulas() {
        let (pressure, lat, height) = (1013.25, core::f64::consts::FRAC_PI_4, 0.0);
        let ztd = saastamoinen_zenith_delay(pressure, lat, height);
        assert_eq!(ztd, gnss_saastamoinen(pressure, lat, height));
        assert!((2.28..2.32).contains(&ztd), "{ztd} m at sea level");
        // Hydrostatic delay is proportional to surface pressure.
        let half = saastamoinen_zenith_delay(pressure / 2.0, lat, height);
        assert!(
            (half / ztd - 0.5).abs() < 1e-3,
            "halving the pressure gave {half} against {ztd}"
        );

        // (1602.0 + k·0.5625) MHz and (1246.0 + k·0.4375) MHz, over the FDMA channel range.
        for k in -7i8..=6 {
            let (l1, l2) = (glonass_l1_frequency(k), glonass_l2_frequency(k));
            assert_eq!((l1, l2), (gnss_glonass_l1(k), gnss_glonass_l2(k)));
            assert!(
                (l1 - (1602.0 + f64::from(k) * 0.5625) * 1e6).abs() < 1e-6,
                "channel {k} L1 is {l1} Hz"
            );
            assert!(
                (l2 - (1246.0 + f64::from(k) * 0.4375) * 1e6).abs() < 1e-6,
                "channel {k} L2 is {l2} Hz"
            );
        }
        // The L1/L2 ratio is the FDMA 9/7, channel by channel.
        let (l1, l2) = (glonass_l1_frequency(3), glonass_l2_frequency(3));
        assert!((l1 / l2 - 9.0 / 7.0).abs() < 1e-12, "{l1} / {l2}");
    }

    /// The four orbit-design helpers are one-line wrappers over closed forms, so each is pinned by
    /// the condition that defines it rather than by the crate call alone: the Sun-synchronous
    /// inclination has to make the `J2` node rate equal the mean Sun's, the repeat axis has to close
    /// `revs` revolutions in `days`, the frozen eccentricity has to be small and positive, and the
    /// critical inclination is `acos(1/√5)` exactly.
    #[test]
    fn the_orbit_design_helpers_satisfy_the_conditions_that_define_them() {
        let (mu, re, j2) = (3.986_004_418e14, 6_378_137.0, 1.082_626_68e-3);
        let a = re + 700e3;

        let inc =
            sun_synchronous_inclination(a, 0.0, mu, re, j2).expect("700 km is sun-synchronous");
        // Ω̇ = −1.5·n·J2·(Re/a)²·cos i has to equal the tropical-year node rate.
        let n = (mu / a.powi(3)).sqrt();
        let node_rate = -1.5 * n * j2 * (re / a).powi(2) * inc.cos();
        assert!(
            (node_rate / SUN_SYNCHRONOUS_NODE_RATE - 1.0).abs() < 1e-9,
            "the node drifts at {node_rate} rad/s, the Sun at {SUN_SYNCHRONOUS_NODE_RATE}"
        );
        assert!(
            (97.5..98.7).contains(&inc.to_degrees()),
            "{} deg is not a Sun-synchronous inclination",
            inc.to_degrees()
        );
        assert!(
            sun_synchronous_inclination(42_164e3, 0.0, mu, re, j2).is_none(),
            "J2 cannot precess a geosynchronous node a full turn a year"
        );

        // Landsat-8's 233/16 repeat lands near 705 km.
        let rate = 7.292_115_146_7e-5;
        let (revs, days, i8_deg) = (233u32, 16u32, 98.2_f64.to_radians());
        let a_repeat = repeat_ground_track_sma(revs, days, i8_deg, 0.0, mu, re, j2, rate);
        assert!(
            (re + 690e3..re + 720e3).contains(&a_repeat),
            "{a_repeat} m is not the Landsat repeat axis"
        );
        // More revolutions in the same days is a lower orbit -- the repeat condition, not a table.
        let faster = repeat_ground_track_sma(revs + 16, days, i8_deg, 0.0, mu, re, j2, rate);
        assert!(faster < a_repeat, "{faster} m is not below {a_repeat} m");

        let e = frozen_eccentricity(a, 98.0_f64.to_radians(), re, j2, -2.532_653_65e-6);
        assert!((0.8e-3..1.3e-3).contains(&e), "{e} is not a frozen orbit");
        // e_frozen ∝ −J3·sin i / (2·J2·a/Re): doubling |J3| doubles the eccentricity.
        let doubled = frozen_eccentricity(a, 98.0_f64.to_radians(), re, j2, -5.065_307_30e-6);
        assert!((doubled / e - 2.0).abs() < 1e-9, "{doubled} against {e}");

        let crit = critical_inclination();
        assert!(
            (crit.cos() - (1.0f64 / 5.0).sqrt()).abs() < 1e-15,
            "{crit} rad is not acos(1/√5)"
        );
        assert!((crit.to_degrees() - 63.434_948_822_922).abs() < 1e-9);
    }

    /// Both frame bindings unpack plain numbers into epochs and typed angles and then hand the state
    /// to the crate's transform, so the oracle is that transform applied to the same state. What
    /// makes the equality non-vacuous is the physics either side of it: a frame change is a rotation,
    /// so `|r|` survives it exactly; half a sidereal day of UT1 turns the Earth-fixed position
    /// through ~180 deg; polar motion at the arcsecond scale moves the ground point by metres; and
    /// the rigorous CIO chain and the GMST-only TEME chain are kilometres apart on the same numbers,
    /// which is what says the two bindings are not one code path with two names.
    #[test]
    fn the_frame_bindings_rotate_the_state_the_way_their_crate_transforms_do() {
        let (r, v) = ([7_000e3, 1_200e3, -900e3], [1.0e3, 7.0e3, 0.5e3]);
        let (ut1_ns, tt_ns) = (631_152_000_000_000_000_i128, 631_152_069_184_000_000_i128);
        let (xp, yp, lod) = (1.2e-6, -0.8e-6, 1e-3);

        let (r_teme, v_teme) = teme_to_itrf(r, v, ut1_ns, tt_ns, xp, yp, lod);
        let tf = frames_teme_to_itrf(
            Epoch::<Ut1>::from_ns_since_j2000(ut1_ns),
            Epoch::<Tt>::from_ns_since_j2000(tt_ns),
            Angle::from_radians(xp),
            Angle::from_radians(yp),
            lod,
        );
        let sv = tf.state(StateVector::<Teme>::new(vec3(r), vec3(v)));
        assert_eq!(
            (r_teme, v_teme),
            (arr(sv.position.vector), arr(sv.velocity.vector)),
            "teme_to_itrf is not the crate transform"
        );

        let (r_gcrf, v_gcrf) = gcrf_to_itrf(r, v, ut1_ns, tt_ns, xp, yp, 1e-9, -2e-9, lod);
        let tf = frames_gcrf_to_itrf(
            Epoch::<Ut1>::from_ns_since_j2000(ut1_ns),
            Epoch::<Tt>::from_ns_since_j2000(tt_ns),
            Angle::from_radians(xp),
            Angle::from_radians(yp),
            Angle::from_radians(1e-9),
            Angle::from_radians(-2e-9),
            lod,
        );
        let sv = tf.state(StateVector::<Gcrf>::new(vec3(r), vec3(v)));
        assert_eq!(
            (r_gcrf, v_gcrf),
            (arr(sv.position.vector), arr(sv.velocity.vector)),
            "gcrf_to_itrf is not the crate transform"
        );

        let norm = |p: [f64; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        for (name, out) in [("teme", r_teme), ("gcrf", r_gcrf)] {
            assert!(
                (norm(out) - norm(r)).abs() < 1e-6,
                "{name} changed |r| from {} to {}",
                norm(r),
                norm(out)
            );
        }
        // The two chains disagree by the precession-nutation the TEME one approximates away.
        let apart = norm([
            r_teme[0] - r_gcrf[0],
            r_teme[1] - r_gcrf[1],
            r_teme[2] - r_gcrf[2],
        ]);
        assert!(apart > 1e3, "the two frame chains landed {apart} m apart");

        // Half a sidereal day of UT1 turns the Earth-fixed position most of the way round: the mean
        // sidereal day is 86 164.090 530 833 s, so half of it is 43 082.045 265 417 s.
        let half_day_ns = 43_082_045_265_417_i128;
        let (turned, _) = teme_to_itrf(r, v, ut1_ns + half_day_ns, tt_ns, xp, yp, lod);
        let dot = turned[0] * r_teme[0] + turned[1] * r_teme[1] + turned[2] * r_teme[2];
        assert!(
            dot < -0.5 * norm(r) * norm(r),
            "half a sidereal day did not turn the frame: cos = {}",
            dot / (norm(r) * norm(r))
        );

        // 0.25 arcsec of polar motion is metres on the ground, and it has to reach the result.
        let (unpolar, _) = teme_to_itrf(r, v, ut1_ns, tt_ns, 0.0, 0.0, lod);
        let shift = norm([
            unpolar[0] - r_teme[0],
            unpolar[1] - r_teme[1],
            unpolar[2] - r_teme[2],
        ]);
        assert!(
            (1.0..100.0).contains(&shift),
            "polar motion moved {shift} m"
        );
    }

    /// The phase-unwrapping family. A linear ramp that runs several times past ±π is wrapped and
    /// handed to all four unwrappers: each has to put it back, up to the single constant offset
    /// unwrapping cannot recover. That is a constructed truth, not a replay of the crate call --
    /// and `residues` has to report a clean field, because a smooth ramp has no residue anywhere.
    #[test]
    fn the_phase_unwrapping_bindings_recover_the_ramp_they_were_given() {
        let (rows, cols) = (8_usize, 12_usize);
        let truth: Vec<f64> = (0..rows * cols)
            .map(|k| 0.9 * (k / cols) as f64 + 1.3 * (k % cols) as f64)
            .collect();
        let wrapped: Vec<f64> = truth.iter().map(|&p| wrap_phase(p)).collect();

        let pi = core::f64::consts::PI;
        assert_eq!(wrap_phase(3.0 * pi), sig_wrap_phase(3.0 * pi));
        for (&w, &p) in wrapped.iter().zip(&truth) {
            assert!((-pi..=pi).contains(&w), "{w} is not wrapped");
            let turns = (w - p) / (2.0 * pi);
            assert!(
                (turns - turns.round()).abs() < 1e-9,
                "{w} is not {p} modulo 2π"
            );
        }

        // A ramp is residue-free, so every charge is zero -- and there is one per 2x2 cell.
        let charges = residues(wrapped.clone(), rows, cols);
        assert_eq!(
            charges,
            sig_residues(&wrapped, rows, cols)
                .into_iter()
                .map(i32::from)
                .collect::<Vec<_>>()
        );
        assert_eq!(charges.len(), (rows - 1) * (cols - 1));
        assert!(
            charges.iter().all(|&c| c == 0),
            "a smooth ramp has residues"
        );

        let row: Vec<f64> = wrapped[..cols].to_vec();
        let got_1d = unwrap_phase_1d(row.clone());
        assert_eq!(got_1d, sig_unwrap_1d(&row));
        for (j, &g) in got_1d.iter().enumerate() {
            let want = truth[j] - truth[0] + got_1d[0];
            assert!(
                (g - want).abs() < 1e-9,
                "1-D column {j}: {g} against {want}"
            );
        }

        for (name, got) in [
            ("path", unwrap_phase_2d(wrapped.clone(), rows, cols)),
            ("lstsq", unwrap_phase_2d_lstsq(wrapped.clone(), rows, cols)),
            (
                "branch-cut",
                unwrap_phase_2d_branch_cut(wrapped.clone(), rows, cols),
            ),
        ] {
            assert_eq!(got.len(), truth.len(), "{name} changed the field size");
            for (k, &g) in got.iter().enumerate() {
                let want = truth[k] - truth[0] + got[0];
                assert!((g - want).abs() < 1e-6, "{name} at {k}: {g} against {want}");
            }
        }
    }

    /// Coherence and the two SAR impulse-response metrics. Coherence is 1 for an image against
    /// itself by construction; the resolution and sidelobe numbers are hand-derived from the shapes
    /// fed in -- `4 − 2√2` samples for a triangular peak, and `20·log10(1/4)` dB for a main lobe four
    /// times its highest sidelobe -- so a constant return has no value it can hide behind.
    #[test]
    fn the_insar_and_impulse_response_metrics_are_the_numbers_their_shapes_imply() {
        let (rows, cols) = (4_usize, 4_usize);
        let n = rows * cols;
        let (re, im): (Vec<f64>, Vec<f64>) = (0..n)
            .map(|k| {
                let phi = 0.3 * k as f64;
                (1.0 + 0.1 * k as f64) * phi.cos()
            })
            .zip((0..n).map(|k| {
                let phi = 0.3 * k as f64;
                (1.0 + 0.1 * k as f64) * phi.sin()
            }))
            .unzip();
        let gamma = coherence(
            re.clone(),
            im.clone(),
            re.clone(),
            im.clone(),
            rows,
            cols,
            3,
        );
        assert_eq!(gamma.len(), n, "coherence dropped pixels");
        for (k, &g) in gamma.iter().enumerate() {
            assert!(
                (g - 1.0).abs() < 1e-12,
                "pixel {k} against itself has coherence {g}"
            );
        }
        // Rotating the second image by a constant phase leaves the magnitude of the average alone.
        let (re2, im2): (Vec<f64>, Vec<f64>) = re
            .iter()
            .zip(&im)
            .map(|(&a, &b)| (a * 0.6 - b * 0.8, a * 0.8 + b * 0.6))
            .unzip();
        let rotated = coherence(re, im, re2, im2, rows, cols, 3);
        for (k, (&g, &r)) in gamma.iter().zip(&rotated).enumerate() {
            assert!((g - r).abs() < 1e-12, "pixel {k}: {g} against {r}");
        }

        // A triangular peak crosses −3 dB (amplitude 1/√2) at √2 and 4 − √2, so it is 4 − 2√2 wide.
        let triangle = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let width = resolution_3db_samples(triangle.clone());
        assert_eq!(width, sig_res3db(&triangle));
        assert!(
            (width - (4.0 - 2.0 * core::f64::consts::SQRT_2)).abs() < 1e-12,
            "{width} samples against 4 − 2√2"
        );
        assert!(
            resolution_3db_samples(Vec::new()).is_nan(),
            "an empty response has no width"
        );

        // Main lobe 1.0, highest sidelobe 0.25 outside it: 20·log10(1/4) = −12.0412 dB.
        let response = vec![0.1, 0.25, 0.1, 0.5, 1.0, 0.5, 0.1, 0.25, 0.1];
        let pslr = peak_sidelobe_ratio_db(response.clone());
        assert_eq!(pslr, sig_pslr(&response));
        assert!(
            (pslr - 20.0 * 0.25_f64.log10()).abs() < 1e-12,
            "{pslr} dB against 20·log10(1/4)"
        );
        assert!(
            peak_sidelobe_ratio_db(Vec::new()).is_nan(),
            "an empty response has no peak"
        );
    }

    /// The three Range-Doppler acquisition parameters, each against the closed form in its own doc
    /// comment: `K_a = 2V²/(λ·r₀)`, the hyperbolic range history behind `ΔR(f_η)`, and the SRC rate
    /// that is undefined at zero Doppler.
    #[test]
    fn the_range_doppler_parameters_are_their_documented_closed_forms() {
        let (prf, v, lambda, r0, spacing) = (1_650.0, 7_100.0, 0.055, 850e3, 4.5);
        let ka = azimuth_fm_rate(prf, v, lambda, r0, spacing);
        assert!(
            (ka - 2.0 * v * v / (lambda * r0)).abs() < 1e-9,
            "{ka} Hz/s is not 2V²/(λ·r₀)"
        );

        // ΔR = λ²·r₀·f_η²/(8V²), the parabolic expansion of the hyperbolic range history: zero at
        // zero Doppler, even in f_η, and quadratic in it.
        let f_eta = 400.0;
        let dr = range_migration(f_eta, prf, v, lambda, r0, spacing);
        assert!(
            (dr - lambda * lambda * r0 * f_eta * f_eta / (8.0 * v * v)).abs() < 1e-12,
            "{dr} m is not λ²·r₀·f_η²/(8V²)"
        );
        assert!(
            (range_migration(2.0 * f_eta, prf, v, lambda, r0, spacing) / dr - 4.0).abs() < 1e-9,
            "the migration is not quadratic in f_η"
        );
        assert!(
            range_migration(0.0, prf, v, lambda, r0, spacing).abs() < 1e-12,
            "zero Doppler migrates nothing"
        );
        assert!(
            range_migration(-f_eta, prf, v, lambda, r0, spacing) > 0.0,
            "migration is even in f_η and positive"
        );

        assert!(
            src_fm_rate(0.0, prf, v, lambda, r0, spacing).is_none(),
            "the SRC rate is undefined at zero Doppler"
        );
        let src = src_fm_rate(f_eta, prf, v, lambda, r0, spacing)
            .expect("a non-zero Doppler has an SRC rate");
        assert!(
            src.is_finite() && src.abs() > 1.0,
            "{src} Hz/s is not an SRC rate"
        );
    }

    /// The two TomoSAR spectra on a covariance built here from a single scatterer: `R = a·aᴴ` at a
    /// known elevation, so both estimators have to peak on that grid node. Length plus peak position
    /// is what neither an empty nor a constant return can produce.
    #[test]
    fn the_tomographic_spectra_peak_on_the_scatterer_their_covariance_holds() {
        let (wavelength, range, truth) = (0.056, 700e3, 40.0);
        let baselines = vec![-100.0, -66.0, -33.0, 0.0, 33.0, 66.0, 100.0];
        let steering: Vec<(f64, f64)> = baselines
            .iter()
            .map(|b| {
                let phi = -4.0 * core::f64::consts::PI * b * truth / (wavelength * range);
                (phi.cos(), phi.sin())
            })
            .collect();
        let n = baselines.len();
        let (mut cov_re, mut cov_im) = (vec![0.0; n * n], vec![0.0; n * n]);
        for i in 0..n {
            for j in 0..n {
                // a_i · conj(a_j)
                cov_re[i * n + j] = steering[i].0 * steering[j].0 + steering[i].1 * steering[j].1;
                cov_im[i * n + j] = steering[i].1 * steering[j].0 - steering[i].0 * steering[j].1;
            }
            cov_re[i * n + i] += 0.05; // the noise floor Capon needs to be invertible
        }
        let elevations: Vec<f64> = (-40..=40).map(|s| f64::from(s) * 5.0).collect();

        let peak_of = |spectrum: &[f64]| {
            spectrum
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .expect("a non-empty spectrum")
                .0
        };
        let bartlett = bartlett_spectrum(
            cov_re.clone(),
            cov_im.clone(),
            baselines.clone(),
            wavelength,
            range,
            elevations.clone(),
        );
        assert_eq!(bartlett.len(), elevations.len());
        assert_eq!(elevations[peak_of(&bartlett)], truth, "Bartlett missed");

        let capon = capon_spectrum(
            cov_re,
            cov_im,
            baselines,
            wavelength,
            range,
            elevations.clone(),
            0.01,
        );
        assert_eq!(capon.len(), elevations.len());
        assert_eq!(elevations[peak_of(&capon)], truth, "Capon missed");
    }

    /// Persistent-scatterer selection: amplitude dispersion is `σ/μ` down the stack, so a pixel held
    /// constant across every acquisition has dispersion 0 and a pixel that swings has a large one.
    /// The threshold then has to keep exactly the stable ones, in index order.
    #[test]
    fn the_persistent_scatterer_bindings_rank_pixels_by_their_own_dispersion() {
        let pixels = 4;
        let stack = vec![
            vec![10.0, 10.0, 1.0, 5.0],
            vec![10.0, 12.0, 9.0, 5.0],
            vec![10.0, 8.0, 5.0, 5.0],
            vec![10.0, 10.0, 3.0, 5.0],
        ];
        let dispersion = amplitude_dispersion(stack, pixels);
        assert_eq!(dispersion.len(), pixels);
        assert!(
            dispersion[0].abs() < 1e-12 && dispersion[3].abs() < 1e-12,
            "a constant pixel disperses: {dispersion:?}"
        );
        assert!(
            dispersion[2] > dispersion[1] && dispersion[1] > 0.0,
            "the swinging pixel is not the noisiest: {dispersion:?}"
        );

        let kept = select_persistent_scatterers(dispersion.clone(), 0.2);
        assert_eq!(kept, vec![0, 1, 3], "kept {kept:?} from {dispersion:?}");
        assert!(select_persistent_scatterers(dispersion, -1.0).is_empty());
    }

    /// The 6S coupling and the three MODIS RTLSR kernels. The coupling is an exact involution --
    /// surface to TOA and back is the identity, which no constant survives -- and the kernels are
    /// pinned against the independent numpy reference the earthvision unit tests carry.
    #[test]
    fn the_atmospheric_and_brdf_bindings_invert_and_match_their_numpy_reference() {
        let (path, t_down, t_up, albedo) = (0.08, 0.85, 0.88, 0.12);
        let surface = 0.3;
        let toa = atmospheric_toa_from_surface(path, t_down, t_up, albedo, surface);
        assert!(toa > path, "{toa} is below the path reflectance alone");
        assert!(
            (atmospheric_surface_from_toa(path, t_down, t_up, albedo, toa) - surface).abs() < 1e-12,
            "the 6S coupling is not an involution: {toa}"
        );
        // With no atmosphere between them the two reflectances are the same number.
        assert!(
            (atmospheric_toa_from_surface(0.0, 1.0, 1.0, 0.0, surface) - surface).abs() < 1e-15
        );

        let d = f64::to_radians;
        let ross = ross_thick_kernel(d(45.0), d(30.0), d(60.0));
        assert!(
            (ross - 0.061_238_612_477_587_55).abs() < 1e-12,
            "RossThick gave {ross}"
        );
        let li = li_sparse_reciprocal_kernel(d(45.0), d(30.0), d(60.0));
        assert!(
            (li + 0.955_216_045_592_488_1).abs() < 1e-12,
            "LiSparse-R gave {li}"
        );
        // BRDF is the linear combination of the two kernels on the isotropic term.
        let (f_iso, f_vol, f_geo) = (0.15, 0.05, 0.02);
        let brdf = brdf_reflectance(f_iso, f_vol, f_geo, d(45.0), d(30.0), d(60.0));
        assert!(
            (brdf - (f_iso + f_vol * ross + f_geo * li)).abs() < 1e-12,
            "{brdf} is not f_iso + f_vol·K_vol + f_geo·K_geo"
        );
        // At nadir both kernels vanish and only the isotropic term is left.
        assert!((brdf_reflectance(f_iso, f_vol, f_geo, 0.0, 0.0, 0.0) - f_iso).abs() < 1e-12);
    }

    /// A great-circle step: zero arc stays put, due north from the equator raises the latitude by
    /// exactly the central angle, and due east along the equator raises the longitude by it. Three
    /// closed-form cases, so no constant pair can pass them together.
    #[test]
    fn the_ground_point_binding_walks_a_great_circle() {
        let (lat0, lon0) = (0.35, -1.1);
        let (half_pi, arc) = (core::f64::consts::FRAC_PI_2, 0.2);

        let stay = ground_point_at(lat0, lon0, 0.0, 1.0);
        assert_eq!(stay, ev_ground_point_at(lat0, lon0, 0.0, 1.0));
        assert!(
            (stay.0 - lat0).abs() < 1e-12 && (stay.1 - lon0).abs() < 1e-12,
            "a zero-length arc moved to {stay:?}"
        );

        let north = ground_point_at(0.0, lon0, arc, 0.0);
        assert!(
            (north.0 - arc).abs() < 1e-12 && (north.1 - lon0).abs() < 1e-12,
            "due north from the equator landed at {north:?}"
        );

        let east = ground_point_at(0.0, lon0, arc, half_pi);
        assert!(
            east.0.abs() < 1e-12 && (east.1 - (lon0 + arc)).abs() < 1e-12,
            "due east along the equator landed at {east:?}"
        );
    }

    /// Detection post-processing. The IoU of two half-overlapping unit boxes is a number the test
    /// works out itself, and non-maximum suppression has to keep the best box of each overlapping
    /// group in score order -- so both the ordering and the threshold are pinned.
    #[test]
    fn the_detection_bindings_suppress_by_the_overlap_they_measure() {
        // Two 2x2 boxes offset by 1 in x: intersection 1x2 = 2, union 4 + 4 − 2 = 6.
        let iou = box_iou((0.0, 0.0, 2.0, 2.0), (1.0, 0.0, 3.0, 2.0));
        assert!((iou - 2.0 / 6.0).abs() < 1e-12, "{iou} is not 1/3");
        assert!(
            (box_iou((0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 2.0, 2.0)) - 1.0).abs() < 1e-12,
            "a box does not overlap itself"
        );
        assert!(
            box_iou((0.0, 0.0, 1.0, 1.0), (5.0, 5.0, 6.0, 6.0)).abs() < 1e-12,
            "disjoint boxes overlap"
        );

        // Index 1 is the best of the overlapping pair {0, 1}; index 2 is far away and survives.
        let boxes = vec![
            (0.0, 0.0, 2.0, 2.0),
            (0.1, 0.1, 2.1, 2.1),
            (10.0, 10.0, 12.0, 12.0),
        ];
        let scores = vec![0.4, 0.9, 0.7];
        assert_eq!(
            nms(boxes.clone(), scores.clone(), 0.5),
            vec![1, 2],
            "the wrong boxes survived suppression"
        );
        // A threshold above every overlap suppresses nothing, in descending score order.
        assert_eq!(nms(boxes, scores, 1.1), vec![1, 2, 0]);
    }

    /// Both pan-sharpeners substitute an intensity component, so a panchromatic band that already
    /// *is* that component has to leave the multispectral bands where they were. The component is
    /// not the same for the two: Brovey normalises by the band **sum**, Gram-Schmidt simulates its
    /// low-resolution pan as the band **mean**. Feeding each its own fixed point is a constructed
    /// identity rather than a stored image, and it pins the band count, the pixel count and the
    /// scaling at once.
    #[test]
    fn the_pan_sharpen_bindings_leave_their_own_intensity_component_alone() {
        let bands = vec![
            vec![0.20, 0.40, 0.60, 0.80],
            vec![0.30, 0.35, 0.45, 0.55],
            vec![0.10, 0.25, 0.30, 0.45],
        ];
        let intensity: Vec<f64> = (0..4).map(|k| bands.iter().map(|b| b[k]).sum()).collect();
        let mean: Vec<f64> = intensity.iter().map(|s| s / bands.len() as f64).collect();

        for (name, sharpened) in [
            (
                "brovey",
                brovey_pan_sharpen(bands.clone(), intensity.clone()),
            ),
            (
                "gram-schmidt",
                gram_schmidt_pan_sharpen(bands.clone(), mean),
            ),
        ] {
            assert_eq!(sharpened.len(), bands.len(), "{name} lost a band");
            for (b, (got, want)) in sharpened.iter().zip(&bands).enumerate() {
                assert_eq!(got.len(), want.len(), "{name} band {b} changed length");
                for (k, (&g, &w)) in got.iter().zip(want).enumerate() {
                    assert!(
                        (g - w).abs() < 1e-12,
                        "{name} band {b} pixel {k}: {g} against {w}"
                    );
                }
            }
        }

        // Brovey is linear in the panchromatic band: twice the pan is twice every output pixel.
        let brighter: Vec<f64> = intensity.iter().map(|i| i * 2.0).collect();
        let boosted = brovey_pan_sharpen(bands.clone(), brighter);
        for (b, (got, want)) in boosted.iter().zip(&bands).enumerate() {
            for (k, (&g, &w)) in got.iter().zip(want).enumerate() {
                assert!(
                    (g - 2.0 * w).abs() < 1e-12,
                    "brovey band {b} pixel {k}: {g} is not twice {w}"
                );
            }
        }
    }

    /// Semi-global matching on a stereo pair built here: a random-texture left image and the same
    /// texture shifted by a known constant disparity. The recovered field has to be that disparity
    /// over the interior, at full image size -- a length no truncated or constant return matches.
    /// Plane-sweep MVS is checked on the weaker contract it actually promises: one depth per pixel,
    /// drawn from the candidate set it was handed.
    #[test]
    fn the_dense_stereo_bindings_recover_the_disparity_they_were_shown() {
        let (h, w, d0) = (24_usize, 40_usize, 4_usize);
        // A deterministic LCG: the texture has to be random enough to match on, and reproducible.
        let mut seed = 12_345_u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 40) as f64 / f64::from(1_u32 << 24)
        };
        let left: Vec<f64> = (0..h * w).map(|_| next()).collect();
        let right: Vec<f64> = (0..h * w)
            .map(|k| {
                let (y, x) = (k / w, k % w);
                if x + d0 < w {
                    left[y * w + x + d0]
                } else {
                    next()
                }
            })
            .collect();

        let disparity = semi_global_match(left.clone(), right, h, w, 12, 2, 6.0, 60.0);
        assert_eq!(disparity.len(), h * w, "the disparity field changed size");
        let (mut hits, mut total) = (0_usize, 0_usize);
        for y in 3..h - 3 {
            for x in d0 + 3..w - 3 {
                total += 1;
                if (disparity[y * w + x].round() as usize) == d0 {
                    hits += 1;
                }
            }
        }
        assert!(
            hits * 10 > total * 9,
            "{hits} of {total} interior pixels found disparity {d0}"
        );

        // Plane sweep: two pinhole views of the same image, and a depth grid to choose from.
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let ref_cam = PyCamera::new(
            eye,
            [0.0, 0.0, 0.0],
            500.0,
            [w as f64 / 2.0, h as f64 / 2.0],
        );
        let nbr_cam = PyCamera::new(
            eye,
            [2.0, 0.0, 0.0],
            500.0,
            [w as f64 / 2.0, h as f64 / 2.0],
        );
        let depths = vec![50.0, 100.0, 200.0, 400.0];
        let swept = plane_sweep_mvs(
            left.clone(),
            &ref_cam,
            left,
            &nbr_cam,
            h,
            w,
            depths.clone(),
            2,
        );
        assert_eq!(swept.len(), h * w, "the depth map changed size");
        // Every pixel is either a candidate depth or the 0.0 the sweep leaves where no patch was
        // fully inside both views, and the interior has to be mostly the former.
        let mut chosen = 0_usize;
        for (k, d) in swept.iter().enumerate() {
            assert!(
                depths.contains(d) || *d == 0.0,
                "pixel {k} got depth {d}, which is not a candidate"
            );
            chosen += usize::from(*d != 0.0);
        }
        assert!(
            chosen * 2 > h * w,
            "only {chosen} of {} pixels got a depth",
            h * w
        );
    }

    /// SBAS inversion on a scatterer built from the model's own design columns: phases generated
    /// from a chosen line-of-sight velocity and DEM error have to come back as that pair, and
    /// baselines that cannot separate the two columns are the documented `(0, 0)`.
    #[test]
    fn the_sbas_inversion_returns_the_velocity_and_dem_error_it_was_built_from() {
        let (wavelength, range, incidence) = (0.056, 800e3, 0.6);
        let (velocity, dem_error) = (-0.023, 14.5);
        let temporal = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let perp = vec![0.0, 120.0, -80.0, 200.0, -150.0, 60.0];

        let four_pi = 4.0 * core::f64::consts::PI;
        let (kv, kh) = (
            four_pi / wavelength,
            four_pi / (wavelength * range * incidence.sin()),
        );
        let phases: Vec<f64> = temporal
            .iter()
            .zip(&perp)
            .map(|(&t, &b)| kv * t * velocity + kh * b * dem_error)
            .collect();

        let (v, d) = sbas_invert(
            phases.clone(),
            temporal.clone(),
            perp.clone(),
            wavelength,
            range,
            incidence,
        );
        assert!(
            (v - velocity).abs() < 1e-9 && (d - dem_error).abs() < 1e-6,
            "inverted ({v}, {d}) against ({velocity}, {dem_error})"
        );
        assert_eq!(
            (v, d),
            sig_sbas_invert(&phases, &temporal, &perp, wavelength, range, incidence)
        );

        // All-zero baselines leave the normal matrix singular, which the model reports as (0, 0).
        let zeros = vec![0.0; temporal.len()];
        assert_eq!(
            sbas_invert(phases, zeros.clone(), zeros, wavelength, range, incidence),
            (0.0, 0.0)
        );
    }

    /// Two thin physics wrappers with closed forms in their own doc comments: the Sentman molecular
    /// speed ratio `s = v/√(2RT/M)`, and the nonlinear density-regression driver row
    /// `[(F+F81)/2, Ap, ((F+F81)/2)², √Ap, 1]`. Both are checked against the formula, so neither a
    /// constant nor a permuted feature order survives.
    #[test]
    fn the_speed_ratio_and_driver_row_are_the_formulas_they_document() {
        let (speed, temperature, molar_mass) = (7_600.0, 950.0, 0.016);
        let s = molecular_speed_ratio(speed, temperature, molar_mass);
        let gas_constant = 8.314_462_618_153_24; // CODATA 2018 R, J/(mol·K)
        assert!(
            (s - speed / (2.0 * gas_constant * temperature / molar_mass).sqrt()).abs() < 1e-9,
            "{s} is not v/√(2RT/M)"
        );
        // Hotter gas means slower flow relative to thermal motion: s falls as √T rises.
        let hotter = molecular_speed_ratio(speed, 4.0 * temperature, molar_mass);
        assert!((hotter / s - 0.5).abs() < 1e-12, "{hotter} against {s}/2");

        let (f107, f107_81, ap) = (142.0, 128.0, 27.0);
        let row = space_weather_drivers(f107, f107_81, ap);
        let mean = 0.5 * (f107 + f107_81);
        assert_eq!(row.len(), 5, "the driver row is five features");
        for (k, (&got, want)) in row
            .iter()
            .zip([mean, ap, mean * mean, ap.sqrt(), 1.0])
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-12,
                "feature {k}: {got} against {want}"
            );
        }
    }

    /// The full time-variable gravity record off a hand-written ICGEM excerpt: the binding has to
    /// find the coefficient it was given, evaluate its trend and annual terms at the epoch asked
    /// for, report `None` for a degree/order the model does not carry, and count what it parsed.
    #[test]
    fn the_time_variable_gravity_binding_evaluates_the_model_it_parsed() {
        // gfct/trnd/acos/asin for (2,0) and a second, trend-only (3,1), with a reference epoch of
        // 2008.0, in the ICGEM column order. Two coefficients, not one: a count that is always the
        // length of the file cannot be told apart from a count that is always one.
        let gfc = "\
begin_of_head
end_of_head
gfct   2    0 -4.841651437908e-04  0.000000000000e+00 1.0e-11 0.0e-11 20080101.0000
trnd   2    0  1.162755000000e-11  0.000000000000e+00 1.0e-13 0.0e-13
acos   2    0  2.000000000000e-11  0.000000000000e+00 1.0e-13 0.0e-13 1.0
asin   2    0 -3.000000000000e-11  0.000000000000e+00 1.0e-13 0.0e-13 1.0
gfct   3    1  2.190800000000e-06  2.680000000000e-07 1.0e-11 1.0e-11 20080101.0000
trnd   3    1  4.000000000000e-12 -2.000000000000e-12 1.0e-13 1.0e-13
";
        let field = PyTimeVariableGravity::parse(gfc);
        assert_eq!(field.__len__(), 2, "two time-variable coefficients");
        assert!(
            field.coefficient(4, 2, 2020.0).is_none(),
            "a degree the model does not carry has no entry"
        );
        // The second coefficient carries a trend on both C and S and no annual term at all.
        let (c31_ref, s31_ref) = field
            .coefficient(3, 1, 2008.0)
            .expect("(3,1) is in the model");
        let (c31_next, s31_next) = field
            .coefficient(3, 1, 2009.0)
            .expect("(3,1) is in the model");
        assert!(
            (c31_next - c31_ref - 4.0e-12).abs() < 1e-18
                && (s31_next - s31_ref + 2.0e-12).abs() < 1e-18,
            "a year of trend moved (3,1) by {:?}",
            (c31_next - c31_ref, s31_next - s31_ref)
        );

        // At the reference epoch the annual terms are cos(0) = 1 and sin(0) = 0.
        let (c_ref, s_ref) = field
            .coefficient(2, 0, 2008.0)
            .expect("(2,0) is in the model");
        assert!(
            (c_ref - (-4.841_651_437_908e-4 + 2.0e-11)).abs() < 1e-18,
            "at the epoch C20 is {c_ref}"
        );
        assert!(s_ref.abs() < 1e-18, "S20 is not zero: {s_ref}");

        // One year on, the trend has moved it by exactly one year's worth and the annual has returned.
        let (c_next, _) = field
            .coefficient(2, 0, 2009.0)
            .expect("(2,0) is in the model");
        assert!(
            (c_next - c_ref - 1.162_755e-11).abs() < 1e-18,
            "a year of trend moved C20 by {}",
            c_next - c_ref
        );
    }

    /// #310's atmosphere on its own, rather than through a decade of quadrature. The lifetime scan
    /// only ever flies equatorial circular orbits, so `z` is zero at every node and the co-rotation
    /// never leaves the orbit plane: a dropped `z²` and a reversed rotation are both invisible from
    /// the end-to-end answer, and a decade of drag would absorb either without changing an outcome.
    #[test]
    fn the_decay_atmosphere_is_the_exponential_shell_it_documents() {
        let omega = 7.292_115e-5;
        let air = DecayExponential {
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
        // The two parameters the shell is named by: the density at the reference radius is the
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

        // `ω ẑ × r`, which is `[-ω y, ω x, 0]` and nothing else: it is horizontal, perpendicular to
        // the horizontal projection of the position, of magnitude `ω √(x² + y²)`, and prograde. A
        // reflection satisfies the first three of those and not the fourth, so the vector is pinned
        // as written rather than as a set of properties.
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
}

/// Entry points whose signature carries an interpreter token -- a `PyResult` return or a
/// `Python<'_>` / `Bound<'_>` parameter -- cannot be named from a test binary built with
/// `extension-module`: nothing supplies libpython's symbols to it, so the link fails on
/// `PyErr_SetObject` before a single assertion runs. They are reachable under
/// `--no-default-features --features test-interpreter`, which links libpython and starts an
/// interpreter for the process. See the feature comments in `Cargo.toml`.
#[cfg(all(test, feature = "test-interpreter"))]
mod interpreter_tests {
    use super::*;

    /// Speed of light, m/s — the constant the two differenced observables are stated in.
    const LIGHT: f64 = 299_792_458.0;

    /// #314's multistatic observables against the single-station models they generalise. A bistatic
    /// link whose transmitter and receiver sit at one place is that station's monostatic range and
    /// range rate doubled; an inter-satellite link with one end at rest is the fixed-station pair
    /// exactly; and TDOA and FDOA are two legs differenced. Four identities, one marshalling — and
    /// a swapped or dropped argument satisfies none of them.
    #[test]
    fn the_multistatic_bindings_reduce_to_the_single_station_ones() {
        Python::attach(|_| {
            let sat = [7_000e3, 300e3, -150e3, 40.0, 7_480.0, 90.0];
            let ground = [6_378_137.0, 0.0, 0.0];
            let antipode = [4_460e3, 2_682e3, -3_674e3];
            let remote = [antipode[0], antipode[1], antipode[2], 0.0, 0.0, 0.0];
            let at_rest = (ground, [0.0; 3], 0.0);
            let folded = (ground, [0.0; 3], 2.0);

            let mono = range(sat, ground);
            assert!(
                (6.0e5..2.0e6).contains(&mono),
                "the monostatic range is {mono}"
            );
            let two_way = bistatic_range(sat, at_rest, 1.0, folded).expect("the legs are causal");
            assert!(
                (two_way - 2.0 * mono).abs() < 1e-6,
                "{two_way} is not twice {mono}"
            );
            let opening =
                bistatic_range_rate(sat, at_rest, 1.0, folded).expect("the legs are causal");
            assert!(
                (opening - 2.0 * range_rate(sat, ground)).abs() < 1e-9,
                "{opening} is not twice the monostatic rate"
            );
            assert!(
                bistatic_range(sat, at_rest, -1.0, folded).is_err(),
                "a reflection preceding its own transmission was accepted"
            );

            let separation = inter_satellite_range(sat, remote);
            assert!(
                (separation - range(sat, antipode)).abs() < 1e-9,
                "{separation}"
            );
            let closing = inter_satellite_range_rate(sat, remote);
            assert!(
                (closing - range_rate(sat, antipode)).abs() < 1e-9,
                "{closing}"
            );
            // Neither end is privileged: both observables are invariant under swapping them.
            assert!((inter_satellite_range(remote, sat) - separation).abs() < 1e-9);

            let first = (ground, [0.0; 3], 0.05);
            let second = (antipode, [0.0; 3], 0.06);
            let arrival = tdoa(sat, 0.0, first, second).expect("both legs are causal");
            let difference = range(sat, antipode) - range(sat, ground);
            assert!(
                (arrival * LIGHT - difference).abs() < 1e-6,
                "{arrival} s is not {difference} m of path"
            );
            assert!(
                (tdoa(sat, 0.0, second, first).expect("still causal") + arrival).abs() < 1e-15,
                "swapping the terminals must negate the difference"
            );
            let carrier = 8.4e9;
            let rates = range_rate(sat, ground) - range_rate(sat, antipode);
            let shift = fdoa(sat, 0.0, first, second, carrier).expect("both legs are causal");
            assert!((shift - carrier * rates / LIGHT).abs() < 1e-9, "{shift}");
        });
    }

    /// #324's DSN observables. Over a flat exciter the ramp integral is a rectangle, so the range
    /// unit is one product of Moyer's own table-13-1 ratio; the modulus is `2^(C+6)`, so dropping
    /// the lowest ranging component by one halves it; and the averaged Doppler and the measured
    /// frequency of one link sum to the reference term exactly. None of the three survives a
    /// swapped band, a swapped epoch or a dropped turnaround ratio.
    #[test]
    fn the_dsn_bindings_carry_moyers_ratios_and_the_count_identity() {
        Python::attach(|_| {
            let exciter = 7.15e9;
            let flat = vec![(0.0, 1e4, exciter, 0.0)];
            let wide =
                dsn_sequential_range(flat.clone(), "X", 100.0, 101.0, 26).expect("X is a band");
            let unwrapped = (221.0 / (2.0 * 749.0)) * exciter;
            assert!(
                (wide - unwrapped).abs() < 1e-3,
                "{wide} against {unwrapped}"
            );
            let s_band =
                dsn_sequential_range(flat.clone(), "S", 100.0, 101.0, 26).expect("S is a band");
            assert!(
                (s_band / wide - 749.0 / 221.0).abs() < 1e-9,
                "the two uplink denominators give {}",
                s_band / wide
            );
            let narrow =
                dsn_sequential_range(flat.clone(), "S", 100.0, 101.0, 25).expect("S is a band");
            assert!(
                (s_band - narrow - 2.0_f64.powi(31)).abs() < 1e-3,
                "one component lower must wrap by exactly one modulus, not {}",
                s_band - narrow
            );
            assert!(
                dsn_sequential_range(flat.clone(), "L", 100.0, 101.0, 26).is_err(),
                "an unknown band was accepted"
            );

            let turnaround = 880.0 / 749.0;
            let interval = 60.0;
            for rate in [1.0, 1.0 - 1e-5, 1.0 + 3e-6] {
                let counted = dsn_averaged_doppler(
                    flat.clone(),
                    "X",
                    "X",
                    exciter,
                    "X",
                    interval,
                    100.0,
                    100.0 + interval * rate,
                    true,
                )
                .expect("the ramp covers the interval");
                let instantaneous = dsn_measured_frequency(flat.clone(), "X", "X", 100.0, rate)
                    .expect("the ramp covers the epoch");
                assert!(
                    (counted + instantaneous - turnaround * exciter).abs() < 1e-3,
                    "the two observables sum to {}",
                    counted + instantaneous
                );
            }
            assert!(
                dsn_measured_frequency(flat, "X", "X", 1e6, 1.0).is_err(),
                "an epoch outside the ramp table was accepted"
            );
        });
    }

    /// ΔDOR against a quasar and the solar-corona delay every deep-space observable carries. The
    /// quasar sits at infinity, so it contributes the projected baseline over `c` and nothing else;
    /// the corona delay is a group delay, falls as `1/f²`, and has no honest value through the
    /// photosphere.
    #[test]
    fn the_delta_dor_and_corona_bindings_carry_their_own_geometry() {
        Python::attach(|_| {
            let goldstone = [-2_353_621.4, -4_641_341.5, 3_677_052.3];
            let canberra = [-4_460_894.9, 2_682_361.5, -3_674_748.5];
            let first = (goldstone, [0.0; 3], 0.05);
            let second = (canberra, [0.0; 3], 0.06);
            let spacecraft = [1.2e11, 3.0e10, -8.0e9, 0.0, 0.0, 0.0];
            let (right_ascension, declination) = (1.1, -0.4);
            let source = [
                declination.cos() * right_ascension.cos(),
                declination.cos() * right_ascension.sin(),
                declination.sin(),
            ];
            let projected = (0..3)
                .map(|k| (canberra[k] - goldstone[k]) * source[k])
                .sum::<f64>()
                / LIGHT;
            let observed = delta_dor(spacecraft, 0.0, first, second, right_ascension, declination)
                .expect("both legs are causal");
            let arrival = tdoa(spacecraft, 0.0, first, second).expect("both legs are causal");
            assert!(
                (observed - arrival - projected).abs() < 1e-12,
                "the quasar term is {}, not {projected}",
                observed - arrival
            );

            let au = 1.495_978_707e11;
            let solar_radius = 696e6;
            let impact = 10.0 * solar_radius;
            let receiver = [au, 0.0, 0.0];
            let depth = -1.5 * au;
            let offset = impact * (au - depth) / (au * au - impact * impact).sqrt();
            let transmitter = [depth, offset, 0.0];
            let at_x = solar_corona_delay(transmitter, receiver, 8.4e9, true, None)
                .expect("the ray clears the photosphere");
            // 10 solar radii at X-band: tens of metres, which is why a superior conjunction is
            // the one geometry a deep-space link cannot ignore this correction on.
            assert!((1.0..1_000.0).contains(&at_x), "a group delay of {at_x} m");
            let at_half = solar_corona_delay(transmitter, receiver, 4.2e9, true, None)
                .expect("the ray clears the photosphere");
            assert!(
                (at_half / at_x - 4.0).abs() < 1e-9,
                "the ratio is {}",
                at_half / at_x
            );
            let phase = solar_corona_delay(transmitter, receiver, 8.4e9, false, None)
                .expect("the ray clears the photosphere");
            assert!((phase + at_x).abs() < 1e-12, "the phase advance is {phase}");
            let doubled = solar_corona_delay(
                transmitter,
                receiver,
                8.4e9,
                true,
                Some(vec![(2.0e12, 2.0)]),
            )
            .expect("the ray clears the photosphere");
            let single = solar_corona_delay(
                transmitter,
                receiver,
                8.4e9,
                true,
                Some(vec![(1.0e12, 2.0)]),
            )
            .expect("the ray clears the photosphere");
            assert!(
                (doubled / single - 2.0).abs() < 1e-12,
                "the column is not linear"
            );
            assert!(
                solar_corona_delay(
                    [1e11, 0.5 * solar_radius, 0.0],
                    [0.0, 0.5 * solar_radius, 0.0],
                    8.4e9,
                    true,
                    None
                )
                .is_err(),
                "a line of sight inside the Sun was accepted"
            );
        });
    }

    /// #310's lifetime bindings. A 200 km cannonball comes down inside days and a 2 000 km one
    /// outlives any horizon worth propagating, the ensemble's quantiles bracket its own members, and
    /// the log-log sensitivity to the ballistic coefficient is the `-1` a drag-dominated decay has.
    /// The three outcomes are three answers and the binding must not merge them.
    #[test]
    fn the_lifetime_bindings_separate_their_outcomes_and_follow_the_air() {
        Python::attach(|_| {
            let re = 6_378_137.0;
            let mu = 3.986_004_418e14;
            let zonals = vec![0.0, 0.0, -1.082_626_68e-3];
            let low = [re + 200e3, 0.0, 0.0, 0.0, 0.0, 0.0];
            let decay = |field: Vec<f64>, scale: f64, rotation: f64| {
                orbital_lifetime(
                    low,
                    field,
                    2.2,
                    0.02,
                    2e-11,
                    re + 300e3,
                    60e3,
                    40.0 * 86_400.0,
                    86_400.0,
                    3_600.0,
                    scale,
                    mu,
                    re,
                    120e3,
                    rotation,
                )
            };
            let entered = |scale: f64| {
                decay(zonals.clone(), scale, EARTH_ROTATION_RATE)
                    .expect("a 200 km cannonball is propagable")
            };
            let (outcome, seconds, perigee) = entered(1.0);
            assert_eq!(outcome, "reentry");
            assert!(
                (86_400.0..10.0 * 86_400.0).contains(&seconds),
                "re-entry at {seconds} s"
            );
            assert!((100e3..121e3).contains(&perigee), "reported at {perigee} m");
            assert!(
                entered(2.0).1 < seconds,
                "twice the air must come down sooner"
            );
            // The shell co-rotates, so the satellite meets it at `v − ω×r` rather than at `v` --
            // some 480 m/s of the 7.8 km/s at this radius. A frozen atmosphere is the faster
            // headwind and must come down sooner; a lifetime that did not move would mean the
            // rotation rate never reached the drag quadrature at all.
            assert!(
                decay(zonals.clone(), 1.0, 0.0)
                    .expect("a frozen atmosphere is still an atmosphere")
                    .1
                    < seconds,
                "a frozen atmosphere must bring it down sooner than a co-rotating one"
            );
            // `DsstZonal::new` asserts on a list that stops short of J2, and an assertion across the
            // FFI boundary is an abort rather than an exception -- so the length is refused here.
            assert!(
                decay(vec![0.0, 0.0], 1.0, EARTH_ROTATION_RATE).is_err(),
                "a zonal list that stops short of J2 is not a zonal field"
            );

            let (survives, horizon, _) = orbital_lifetime(
                [re + 2_000e3, 0.0, 0.0, 0.0, 0.0, 0.0],
                zonals.clone(),
                2.2,
                0.02,
                2e-11,
                re + 300e3,
                60e3,
                5.0 * 86_400.0,
                86_400.0,
                3_600.0,
                1.0,
                mu,
                re,
                120e3,
                EARTH_ROTATION_RATE,
            )
            .expect("a 2 000 km orbit is propagable");
            assert_eq!(survives, "survives");
            assert!(
                (horizon - 5.0 * 86_400.0).abs() < 1e-9,
                "reported {horizon}"
            );

            let (quantiles, members, survivors, unresolved) = orbital_lifetime_ensemble(
                low,
                zonals.clone(),
                2.2,
                0.02,
                2e-11,
                re + 300e3,
                60e3,
                40.0 * 86_400.0,
                vec![0.5, 1.0, 2.0],
                vec![0.0, 0.5, 1.0],
                86_400.0,
                3_600.0,
                mu,
                re,
                120e3,
                EARTH_ROTATION_RATE,
            )
            .expect("three propagable members");
            assert_eq!((members, survivors, unresolved), (3, 0, 0));
            let ordered: Vec<f64> = quantiles
                .iter()
                .map(|q| q.expect("all re-entered"))
                .collect();
            assert!(
                ordered[0] <= ordered[1] && ordered[1] <= ordered[2],
                "{ordered:?}"
            );
            assert!(
                (ordered[2] - entered(0.5).1).abs() < 1.0,
                "the slowest member is the last quantile"
            );

            let slope = ballistic_log_sensitivity(
                low,
                zonals,
                2.2,
                0.02,
                2e-11,
                re + 300e3,
                60e3,
                40.0 * 86_400.0,
                86_400.0,
                3_600.0,
                0.05,
                1.0,
                mu,
                re,
                120e3,
                EARTH_ROTATION_RATE,
            )
            .expect("both sides re-enter");
            assert!((slope + 1.0).abs() < 0.3, "d ln T / d ln (S/m) = {slope}");
        });
    }

    /// #311's element-set emitter. The two lines are a column layout, so they are pinned as lines —
    /// and the catalog number that the five columns cannot hold is refused rather than truncated,
    /// which is the whole reason the number is checked before the line is built.
    #[test]
    fn the_tle_binding_emits_both_lines_in_their_own_columns() {
        Python::attach(|_| {
            let epoch = (2008, 9, 20, 12, 25, 12.25);
            let parameters = [
                15.721_253_91,
                0.000_670_3,
                51.641_6,
                247.462_7,
                130.536_0,
                325.028_8,
            ];
            let (line1, line2) = tle_lines(
                epoch,
                25544,
                parameters,
                -1.1606e-4,
                Some("1998-067A"),
                292,
                56353,
            )
            .expect("the ISS element set fits its own columns");
            assert_eq!(
                line1,
                "1 25544U 98067A   08264.51750289  .00000000  00000-0 -11606-3 0  2921"
            );
            assert_eq!(
                line2,
                "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
            );
            assert!(
                tle_lines(epoch, 270_123, parameters, 0.0, None, 0, 0).is_err(),
                "a six-digit catalog number was printed into five columns"
            );
        });
    }

    /// A minimal well-formed RINEX 3 observation file, written column by column. Generating it from
    /// the reader would make it agree with the reader's own mistakes.
    fn rinex_obs_fixture(gps_codes: &str) -> String {
        let pad = |body: &str, label: &str| format!("{body:<60}{label}");
        [
            pad(
                "     3.05           OBSERVATION DATA    M: MIXED",
                "RINEX VERSION / TYPE",
            ),
            pad(&format!("G    2 {gps_codes}"), "SYS / # / OBS TYPES"),
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

    /// #312's observation reader. The declared descriptor order is the only thing naming a value, so
    /// reversing it must move the values with it while the data records stay byte for byte the same;
    /// and the per-satellite series is that lookup, with the receiver's own loss-of-lock bit.
    #[test]
    fn the_rinex_observation_bindings_read_values_out_of_their_declared_slots() {
        Python::attach(|_| {
            let text = rinex_obs_fixture("C1C L1C");
            let (header, epochs) = rinex_obs(&text).expect("the fixture is well formed");
            assert!((header.0 - 3.05).abs() < 1e-12, "version {}", header.0);
            assert_eq!(header.1, 'M');
            assert_eq!(header.2, "");
            assert_eq!(header.3, Some([4_460_894.9, 2_682_361.5, -3_674_748.5]));
            assert_eq!(header.4, None);
            assert_eq!(header.5, Some(30.0));
            // No `TIME OF FIRST OBS`, and the format's default is GPS rather than "unknown".
            assert_eq!(header.6, "GPS");
            assert_eq!(
                header.7,
                vec![
                    ('G', vec!["C1C".to_owned(), "L1C".to_owned()]),
                    ('R', vec!["C1C".to_owned()]),
                ]
            );
            assert_eq!(epochs.len(), 2);
            assert!(
                (epochs[1].0 - epochs[0].0 - 30.0).abs() < 1e-9,
                "the epochs are {} s apart",
                epochs[1].0 - epochs[0].0
            );
            assert_eq!((epochs[0].1, epochs[0].2), (false, None));
            assert_eq!(epochs[0].3[0].0, "G07");
            assert_eq!(
                epochs[0].3[0].1,
                vec![
                    Some((23_629_347.915, None, Some(6))),
                    Some((124_144_623.325, None, Some(7))),
                ]
            );
            assert_eq!(epochs[0].3[1].1, vec![Some((21_000_000.0, None, None))]);
            assert_eq!(epochs[1].3[0].1[0], Some((23_629_400.0, Some(1), Some(6))));
            // Nothing about the data records changed, so a reader matching values to descriptors by
            // anything other than position would disagree here.
            let (reversed, moved) = rinex_obs(&rinex_obs_fixture("L1C C1C")).expect("well formed");
            assert_eq!(reversed.7[0].1, vec!["L1C".to_owned(), "C1C".to_owned()]);
            assert_eq!(moved[0].3[0].1, epochs[0].3[0].1);

            let series = rinex_obs_series(&text, "G07", "C1C").expect("G declares C1C");
            assert_eq!(series.len(), 2);
            assert!((series[0].1 - 23_629_347.915).abs() < 1e-9);
            assert!((series[1].1 - 23_629_400.0).abs() < 1e-9);
            assert_eq!((series[0].2, series[1].2), (false, true));
            // An absent satellite is absent, not zero: collapsing the two would put a 21 000 km step
            // into the arc.
            assert_eq!(
                rinex_obs_series(&text, "R05", "C1C")
                    .expect("R declares C1C")
                    .len(),
                1
            );
            assert!(
                rinex_obs_series(&text, "R05", "L1C").is_err(),
                "R declares no L1C"
            );
            // Two satellites of one constellation is the case a lookup on either half of the
            // identifier alone cannot tell apart: G09 sits behind G07 in the same epoch, so a match
            // that stopped at the system letter would report G07's range under G09's name.
            let ninth = rinex_obs_series(&text, "G09", "C1C").expect("G declares C1C");
            assert_eq!(ninth.len(), 1);
            assert!((ninth[0].1 - 23_626_000.0).abs() < 1e-9, "{}", ninth[0].1);
            assert!(
                rinex_obs_series(&text, "07", "C1C").is_err(),
                "a PRN is not a satellite"
            );
            assert!(rinex_obs("nothing of the sort").is_err());
        });
    }

    /// #312's dual-frequency combinations. Each is a claim about what cancels, and injecting the
    /// thing it claims to remove is the way to check it: the ionosphere-free code combination
    /// returns the range it was built from, the geometry-free one returns the dispersion alone, and
    /// Melbourne-Wübbena returns the integer wide-lane ambiguity that went in.
    #[test]
    fn the_carrier_combination_bindings_recover_what_was_injected() {
        Python::attach(|_| {
            let f1 = carrier_frequency('G', '1').expect("GPS L1 is a band");
            let f2 = carrier_frequency('G', '2').expect("GPS L2 is a band");
            assert!((f1 - 1_575.42e6).abs() < 1e-6 && (f2 - 1_227.60e6).abs() < 1e-6);
            // GLONASS L1 is FDMA: the frequency belongs to the channel, not to the band.
            assert_eq!(carrier_frequency('R', '1'), None);
            assert_eq!(carrier_frequency('G', '9'), None);

            let (lambda1, lambda2, wide, narrow) =
                lane_wavelengths(f1, f2).expect("descending and positive");
            assert!((lambda1 - LIGHT / f1).abs() < 1e-15);
            assert!((lambda2 - LIGHT / f2).abs() < 1e-15);
            // The two published GPS L1/L2 lane wavelengths, 0.8619 m and 0.1070 m -- the reason
            // the wide-lane ambiguity is the one an arc resolves first.
            assert!((wide - 0.8619).abs() < 1e-4, "the wide lane is {wide} m");
            assert!(
                (narrow - 0.1070).abs() < 1e-4,
                "the narrow lane is {narrow} m"
            );
            assert!(
                lane_wavelengths(f2, f1).is_err(),
                "an ascending pair was accepted"
            );

            let (rho, iono1) = (22_100_000.0, 4.5);
            let iono2 = iono1 * (f1 / f2) * (f1 / f2);
            let (p1, p2) = (rho + iono1, rho + iono2);
            let (n1, n2) = (5.0, 3.0);
            let (l1, l2) = (rho - iono1 + n1 * lambda1, rho - iono2 + n2 * lambda2);
            assert!(
                (ionosphere_free(f1, f2, p1, p2).expect("a pair") - rho).abs() < 1e-6,
                "the ionosphere did not cancel"
            );
            // The difference of two 22 100 km pseudoranges: the cancellation leaves the
            // dispersion on top of an ulp of the range, which is 4 nm here.
            assert!((geometry_free(p1, p2) - (iono1 - iono2)).abs() < 1e-8);
            for combination in [ionosphere_free, wide_lane, narrow_lane] {
                let common = combination(f1, f2, rho, rho).expect("a pair");
                assert!(
                    (common - rho).abs() < 1e-6,
                    "a common-mode value became {common}"
                );
            }
            let cycles = melbourne_wubbena(f1, f2, p1, p2, l1, l2, true).expect("a pair");
            assert!(
                (cycles - (n1 - n2)).abs() < 1e-6,
                "the wide-lane ambiguity is {cycles}"
            );
            let metres = melbourne_wubbena(f1, f2, p1, p2, l1, l2, false).expect("a pair");
            assert!((metres - (n1 - n2) * wide).abs() < 1e-6, "{metres} m");
            assert!((graphic(p1, l1) - (rho + 0.5 * n1 * lambda1)).abs() < 1e-6);
        });
    }

    /// #312's two cycle-slip detectors, on the series that separates them. The first differences of
    /// a quadratic run from 0.45 to 16.65, so a threshold that never fires on the curve is far above
    /// a jump of 1 — the first-difference test is blind to it by construction, and the order-2 fit
    /// reproduces the curve exactly, so its prediction residual *is* the jump.
    #[test]
    fn the_cycle_slip_bindings_report_the_verdict_each_detector_can_reach() {
        Python::attach(|_| {
            let times: Vec<f64> = (0..20).map(|k| 30.0 * f64::from(k)).collect();
            let curve: Vec<f64> = times.iter().map(|t| 0.0005 * t * t).collect();
            let values: Vec<f64> = curve
                .iter()
                .enumerate()
                .map(|(k, v)| v + if k >= 10 { 1.0 } else { 0.0 })
                .collect();

            let plain = detect_cycle_slips_geometry_free(times.clone(), values.clone(), 20.0, 60.0)
                .expect("a positive threshold and gap");
            assert_eq!(plain[0], ("arc_start", None));
            assert!(
                plain[1..].iter().all(|(tag, _)| *tag == "continuous"),
                "{plain:?}"
            );
            let statistic = plain[1].1.expect("a tested sample carries its statistic");
            assert!(
                (statistic - (curve[1] - curve[0])).abs() < 1e-9,
                "{statistic}"
            );

            let fitted =
                detect_cycle_slips_polynomial(times.clone(), values.clone(), 2, 5, 0.5, 60.0)
                    .expect("a window longer than the order");
            let slipped: Vec<usize> = fitted
                .iter()
                .enumerate()
                .filter(|(_, (tag, _))| *tag == "slipped")
                .map(|(k, _)| k)
                .collect();
            assert_eq!(slipped, vec![10], "{fitted:?}");
            let residual = fitted[10].1.expect("a slip carries its statistic");
            assert!((residual - 1.0).abs() < 1e-6, "the residual is {residual}");
            assert!(
                fitted[..5].iter().all(|(tag, _)| *tag == "arc_start"),
                "{fitted:?}"
            );

            let gapped: Vec<f64> = times
                .iter()
                .enumerate()
                .map(|(k, t)| if k >= 10 { t + 7_200.0 } else { *t })
                .collect();
            let broken = detect_cycle_slips_geometry_free(gapped, values.clone(), 20.0, 60.0)
                .expect("a positive threshold and gap");
            assert_eq!(
                broken[10],
                ("arc_start", None),
                "a long gap must restart the arc"
            );

            assert!(
                detect_cycle_slips_geometry_free(times.clone(), values[..19].to_vec(), 20.0, 60.0)
                    .is_err(),
                "two series of different lengths were accepted"
            );
            assert!(
                detect_cycle_slips_geometry_free(times.clone(), values.clone(), -1.0, 60.0)
                    .is_err(),
                "a threshold that rejects every sample was accepted"
            );
            assert!(
                detect_cycle_slips_polynomial(times, values, 2, 3, 0.5, 60.0).is_err(),
                "a window too short to leave a residual was accepted"
            );
        });
    }

    /// #312/#321's integer least squares. The search solves a transformed problem, so the candidates
    /// have to come back through the transform: the objective recomputed from the returned integers
    /// against the caller's own covariance is what proves they did. `Z` is unimodular, so the
    /// conditional variances still multiply to the determinant they started from.
    #[test]
    fn the_ambiguity_bindings_return_candidates_in_the_callers_parametrisation() {
        Python::attach(|_| {
            let covariance = vec![1.2, 1.1, 1.1, 1.05];
            let float_ambiguities = vec![5.45, -3.10];
            let determinant = covariance[0] * covariance[3] - covariance[1] * covariance[2];
            let inverse = [
                covariance[3] / determinant,
                -covariance[1] / determinant,
                -covariance[2] / determinant,
                covariance[0] / determinant,
            ];
            let objective = |integers: &[i64]| -> f64 {
                #[allow(clippy::cast_precision_loss)]
                let d = [
                    integers[0] as f64 - float_ambiguities[0],
                    integers[1] as f64 - float_ambiguities[1],
                ];
                (0..2)
                    .map(|i| {
                        (0..2)
                            .map(|j| d[i] * inverse[2 * i + j] * d[j])
                            .sum::<f64>()
                    })
                    .sum()
            };

            let candidates = resolve_ambiguities(float_ambiguities.clone(), covariance.clone(), 4)
                .expect("a positive definite covariance");
            assert_eq!(candidates.len(), 4);
            for (integers, squared_norm) in &candidates {
                let expected = objective(integers);
                assert!(
                    (squared_norm - expected).abs() < 1e-9 * expected.max(1.0),
                    "{squared_norm} against {expected} at {integers:?}"
                );
            }
            assert!(
                candidates.windows(2).all(|w| w[0].1 <= w[1].1),
                "the candidates are not in order"
            );

            let (transformation, conditional) =
                ambiguity_decorrelation(float_ambiguities, covariance).expect("positive definite");
            assert!(
                (conditional[0] * conditional[1] - determinant).abs() < 1e-9,
                "the decorrelation changed the volume"
            );
            let unimodular =
                transformation[0] * transformation[3] - transformation[1] * transformation[2];
            assert!(
                (unimodular.abs() - 1.0).abs() < 1e-12,
                "det Z = {unimodular}"
            );

            // With nothing to condition on, bootstrapping is plain rounding -- and that is also the
            // search's own answer there, so the two entry points must agree.
            let diagonal = vec![0.01, 0.0, 0.0, 0.04];
            let (rounded, cost) =
                ambiguity_bootstrap(vec![5.3, -2.8], diagonal.clone()).expect("positive definite");
            assert_eq!(rounded, vec![5, -3]);
            assert!((cost - 10.0).abs() < 1e-9, "the objective is {cost}");
            let best =
                resolve_ambiguities(vec![5.3, -2.8], diagonal, 1).expect("positive definite");
            assert_eq!(best[0].0, rounded);
            assert!((best[0].1 - cost).abs() < 1e-9);

            assert!(
                resolve_ambiguities(vec![1.0, 2.0], vec![1.0, 0.0, 0.0, -1.0], 2).is_err(),
                "a covariance that is not one was factorised"
            );
            assert!(
                ambiguity_bootstrap(vec![1.0], vec![1.0, 0.0, 0.0, 1.0]).is_err(),
                "a covariance of the wrong side was accepted"
            );
            assert!(
                ambiguity_decorrelation(vec![], vec![]).is_err(),
                "an empty problem"
            );
        });
    }

    /// #92/#315/#322's conjunction bindings. Three routes to one encounter-plane integral must agree
    /// and, at zero miss with an isotropic joint covariance, equal the elementary
    /// `1 - exp(-R²/2s²)`; the screen finds approaches whose time, distance and speed are known in
    /// closed form; the domain check clears the rectilinear encounter it is fed; and the exception a
    /// Python sampler raises surfaces as itself rather than as a `ValueError` about NaN.
    #[test]
    fn the_conjunction_bindings_estimate_screen_and_check_their_own_domain() {
        Python::attach(|py| {
            let isotropic = [[1e4, 0.0, 0.0], [0.0, 1e4, 0.0], [0.0, 0.0, 1e4]];
            let velocity = [0.0, 14_000.0, 0.0];
            let offset = [30.0, 200.0, 40.0];
            let (foster, patera, chan, maximum) =
                collision_probability(offset, velocity, isotropic, isotropic, 5.0, 5.0)
                    .expect("a projectable encounter");
            for estimate in [patera, chan] {
                assert!(
                    (estimate - foster).abs() < 0.02 * foster,
                    "{estimate} against {foster}"
                );
            }
            assert!(
                maximum >= foster,
                "the bound {maximum} sits under the estimate"
            );
            let centred = collision_probability([0.0; 3], velocity, isotropic, isotropic, 5.0, 5.0)
                .expect("a projectable encounter");
            let exact = 1.0 - (-(10.0_f64.powi(2)) / (2.0 * 2.0 * 100.0_f64.powi(2))).exp();
            assert!(
                (centred.0 - exact).abs() < 2e-4 * exact,
                "{} against {exact}",
                centred.0
            );
            // Only the sum of the radii is a cross section.
            let split = collision_probability(offset, velocity, isotropic, isotropic, 1.0, 9.0)
                .expect("a projectable encounter");
            assert!((split.0 - foster).abs() < 1e-12);

            let relative = PyModule::from_code(
                py,
                c"import math\n\
                  RATE = 2.0 * math.pi / 600.0\n\
                  def relative(t):\n\
                  \x20   return ([10000.0 * math.sin(RATE * t), 500.0, 0.0],\n\
                  \x20            [10000.0 * RATE * math.cos(RATE * t), 0.0, 0.0])\n\
                  def truncated(t):\n\
                  \x20   if t > 500.0:\n\
                  \x20       raise RuntimeError('no ephemeris past 500 s')\n\
                  \x20   return relative(t)\n",
                c"screen.py",
                c"screen",
            )
            .expect("the sampler compiles");
            let found = closest_approaches(
                &relative.getattr("relative").expect("defined above"),
                10.0,
                1000.0,
                30.0,
                1e-4,
            )
            .expect("a usable scan request");
            assert_eq!(found.len(), 3, "{found:?}");
            for (k, (tca, distance, speed)) in found.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let expected = 300.0 * (k + 1) as f64;
                assert!((tca - expected).abs() < 1e-3, "approach {k} at {tca}");
                assert!((distance - 500.0).abs() < 1e-6, "at {distance} m");
                assert!(
                    (speed - 10_000.0 * core::f64::consts::TAU / 600.0).abs() < 1e-6,
                    "at {speed} m/s"
                );
            }
            let coarse = closest_approaches(
                &relative.getattr("relative").expect("defined above"),
                10.0,
                1000.0,
                400.0,
                1e-4,
            )
            .expect("a usable scan request");
            assert!(
                coarse.len() < found.len(),
                "the step is the resolution: {coarse:?}"
            );
            assert!(
                closest_approaches(
                    &relative.getattr("truncated").expect("defined above"),
                    10.0,
                    1000.0,
                    30.0,
                    1e-4
                )
                .is_err(),
                "the sampler's own failure was swallowed"
            );

            let sampler = PyModule::from_code(
                py,
                c"def rectilinear(t):\n\
                  \x20   return (([30.0, 200.0 + 14000.0 * t, 40.0], [0.0, 14000.0, 0.0]),\n\
                  \x20            [[20000.0, 0.0, 0.0], [0.0, 20000.0, 0.0], [0.0, 0.0, 20000.0]])\n",
                c"domain.py",
                c"domain",
            )
            .expect("the sampler compiles");
            let sample = sampler.getattr("rectilinear").expect("defined above");
            let (flags, indicators, correction) = conjunction_usage_violations(
                offset, velocity, isotropic, isotropic, 5.0, 5.0, 5_400.0, &sample,
            )
            .expect("a positive period and a projectable encounter");
            assert_eq!(flags, (false, false, false, false, false));
            // Fed its own assumption, the check reports no disagreement: the correction is unity and
            // what is left is the bisection's own width, not a modelling gap.
            assert!(
                (correction - 1.0).abs() < 1e-6,
                "the correction is {correction}"
            );
            assert!(indicators.2 < 1e-6, "the disagreement is {}", indicators.2);
            // The offset is `|δT'|/p_min` and the duration `δT/p_min`, both far under their limits
            // here -- and both non-zero, which is what says the period entered where it is documented
            // to and the two indicators are not one another.
            assert!(
                (1e-7..1e-4).contains(&indicators.0),
                "the offset is {}",
                indicators.0
            );
            assert!(
                (1e-6..0.02).contains(&indicators.1),
                "the duration is {}",
                indicators.1
            );
            let (_, halved, _) = conjunction_usage_violations(
                offset, velocity, isotropic, isotropic, 5.0, 5.0, 2_700.0, &sample,
            )
            .expect("a positive period");
            assert!((halved.0 - 2.0 * indicators.0).abs() < 1e-12);
            assert!((halved.1 - 2.0 * indicators.1).abs() < 1e-12);
            assert!(
                conjunction_usage_violations(
                    offset, velocity, isotropic, isotropic, 5.0, 5.0, 0.0, &sample
                )
                .is_err(),
                "a conjunction was assessed against no orbital period"
            );

            // Filter 1 of an all-on-all screen: a certainty one way and nothing the other.
            let re = 6_378_137.0;
            assert!(radially_separated(
                re + 800e3,
                re + 900e3,
                re + 400e3,
                re + 420e3,
                5_000.0
            ));
            assert!(!radially_separated(
                re + 400e3,
                re + 420e3,
                re + 800e3,
                re + 900e3,
                400e3
            ));
            assert!(!radially_separated(
                f64::NAN,
                re + 420e3,
                re + 800e3,
                re + 900e3,
                5_000.0
            ));
        });
    }

    /// #313's one correction that is a function rather than a decorator chain. The two endpoints
    /// enter as a sum, `2GM/c²` is the whole prefactor, and the 500 km zenith leg is the 0.669 mm
    /// `derivations/shapiro_delay.mac` puts against a bigfloat reference.
    #[test]
    fn the_shapiro_binding_is_symmetric_and_linear_in_the_central_mass() {
        Python::attach(|_| {
            let (re, mu) = (6_378_137.0, 3.986_004_418e14);
            let zenith = shapiro_delay(re, re + 500e3, 500e3, mu);
            assert!(
                (1000.0 * zenith - 0.669).abs() < 1e-3,
                "the leg is {} mm",
                1000.0 * zenith
            );
            assert!((shapiro_delay(re + 500e3, re, 500e3, mu) - zenith).abs() < 1e-18);
            assert!((shapiro_delay(re, re + 500e3, 500e3, 2.0 * mu) - 2.0 * zenith).abs() < 1e-18);
            assert!(
                shapiro_delay(re, re + 500e3, 1000e3, mu) > zenith,
                "a grazing leg is longer"
            );
        });
    }

    /// A circular Lattice Flower with `(No, Nso, Nc) = (P, S, F)` is documented to reproduce a
    /// Walker Delta `i: T/P/F`. Two independent generators, one assertion: the state vectors agree
    /// satellite for satellite. That pins both bindings' marshalling at once, and it is the
    /// documented contract rather than a stored table.
    #[test]
    fn the_lattice_flower_binding_reproduces_the_walker_it_documents() {
        let (planes, per_plane, phasing) = (6u32, 4u32, 2u32);
        let (a, inc, mu) = (7_200e3, 0.9, 3.986_004_418e14);
        let lattice = lattice_flower(planes, per_plane, phasing, a, 0.0, inc, 0.0, 0.0, mu);
        let walker = walker_constellation("delta", inc, planes, per_plane, phasing, a, mu)
            .expect("delta is a pattern");

        assert_eq!(lattice.len(), (planes * per_plane) as usize);
        assert_eq!(lattice.len(), walker.len());
        for (i, ((lr, lv), (wr, wv))) in lattice.iter().zip(&walker).enumerate() {
            for k in 0..3 {
                assert!(
                    (lr[k] - wr[k]).abs() < 1e-6 && (lv[k] - wv[k]).abs() < 1e-9,
                    "satellite {i} axis {k}: lattice {:?} against walker {:?}",
                    (lr[k], lv[k]),
                    (wr[k], wv[k])
                );
            }
        }
        assert!(
            walker_constellation("spiral", inc, planes, per_plane, phasing, a, mu).is_err(),
            "an unknown pattern must be rejected, not defaulted"
        );
        // `star` is the other accepted pattern and a different constellation: a Star system spreads
        // its planes over π of right ascension, a Delta over 2π, so the two cannot coincide.
        let star = walker_constellation("star", inc, planes, per_plane, phasing, a, mu)
            .expect("star is a pattern");
        assert_eq!(star.len(), walker.len());
        assert!(
            star.iter()
                .zip(&walker)
                .any(|((sr, _), (wr, _))| (sr[0] - wr[0]).abs() > 1e3),
            "the star pattern produced the delta constellation"
        );

        // An elliptical lattice is a different constellation, not the same one relabelled.
        let necklace = lattice_flower(planes, per_plane, phasing, a, 0.3, inc, 0.0, 0.0, mu);
        assert_eq!(necklace.len(), lattice.len());
        assert!(
            necklace
                .iter()
                .zip(&lattice)
                .any(|((nr, _), (lr, _))| (nr[0] - lr[0]).abs() > 1e3),
            "e = 0.3 produced the circular lattice"
        );
    }

    /// `minimum_walker` returns the cheapest Walker Delta meeting a footprint, so the answer has to
    /// be internally consistent -- `total = planes x per_plane` -- and, generated and measured, its
    /// α-characteristic has to be within the footprint that was asked for. A footprint no
    /// constellation of the allowed size can meet is `None`, not a lie.
    #[test]
    fn the_minimum_walker_is_a_walker_that_meets_the_footprint_it_was_given() {
        let (footprint, inc, a, mu) = (1.2, 1.1, 7_200e3, 3.986_004_418e14);
        let (planes, per_plane, phasing, total) =
            minimum_walker(footprint, inc, a, mu, 1, 12, 200, 20)
                .expect("a 1.2 rad footprint is coverable with at most 12 satellites");
        assert_eq!(total, planes * per_plane, "the total is not the product");
        assert!(
            (4..=12).contains(&total),
            "{total} satellites is outside the search"
        );
        assert!(phasing < planes, "phasing {phasing} is not a Delta phasing");

        let states = walker_constellation("delta", inc, planes, per_plane, phasing, a, mu)
            .expect("delta is a pattern");
        assert_eq!(states.len(), total as usize);
        let alpha = alpha_characteristic(states, mu, 1, 200, 20);
        assert!(
            alpha <= footprint,
            "the winner needs {alpha} rad, more than the {footprint} rad asked for"
        );

        // Nothing covers the sphere with a pinhole footprint from twelve satellites.
        assert!(minimum_walker(0.05, inc, a, mu, 1, 12, 200, 20).is_none());

        // More satellites see more sky, so the α-characteristic has to fall as the constellation
        // grows -- a monotonicity no constant return value can reproduce.
        let small = walker_constellation("delta", inc, 2, 2, 1, a, mu).expect("delta is a pattern");
        let large = walker_constellation("delta", inc, 6, 6, 3, a, mu).expect("delta is a pattern");
        let (alpha_small, alpha_large) = (
            alpha_characteristic(small, mu, 1, 200, 20),
            alpha_characteristic(large, mu, 1, 200, 20),
        );
        assert!(
            alpha_large < alpha_small,
            "36 satellites need {alpha_large} rad against 4 satellites' {alpha_small} rad"
        );
        assert!(
            (0.0..core::f64::consts::PI).contains(&alpha_large),
            "{alpha_large} rad is not a footprint half-angle"
        );
    }

    /// The symplectic two-body bridge against the universal-variable (Stumpff) solution -- a
    /// different algorithm, not a re-run of the same one, so a wrong sign or a wrong power in the
    /// inline inverse-square law has nowhere to hide. Both precision branches and the rejected tag.
    #[test]
    fn the_two_body_bridge_agrees_with_the_universal_variable_solution_in_both_precisions() {
        // Eccentric and inclined on purpose: on a circle |r| is constant, so a mutation that only
        // rescales the acceleration slides the satellite along its own orbit and every radius still
        // checks out.
        let r0 = [7_000e3, 0.0, 0.0];
        let v0 = [0.0, 6.2e3, 3.1e3];
        let (dt, steps) = (2.0, 900);
        let (rk, vk) = kepler(vec3(r0), vec3(v0), MU_EARTH, dt * steps as f64);
        let (rk, vk) = (arr(rk), arr(vk));
        Python::attach(|py| {
            for precision in ["f64", "dd"] {
                let (r, v) = propagate_two_body(py, r0, v0, MU_EARTH, dt, steps, precision)
                    .expect("both precision tags are accepted");
                // Measured residual against the Stumpff solution: 3.1e-7 m / 5.5e-10 m/s in `f64`,
                // 2.1e-8 m / 3.3e-11 m/s in double-double. That floor is the two algorithms
                // disagreeing, not either of them being wrong; a flipped operator in the inline
                // gravity moves the endpoint by kilometres.
                for i in 0..3 {
                    assert!(
                        (r[i] - rk[i]).abs() < 1e-6 && (v[i] - vk[i]).abs() < 2e-9,
                        "{precision} component {i}: {:?} against Kepler's {rk:?} / {vk:?}",
                        (r, v)
                    );
                }
            }
            assert!(
                propagate_two_body(py, r0, v0, MU_EARTH, dt, steps, "quad").is_err(),
                "an unknown precision tag must be rejected, not silently defaulted"
            );
        });
    }

    /// The J2 bridge against the secular nodal regression, a closed form that owes nothing to the
    /// integrator, plus the one measurement that separates the two precision arms from each other:
    /// Yoshida8 is time-symmetric, so propagating forward and then back with `-dt` returns to the
    /// start up to round-off, and round-off is exactly what double-double buys.
    #[test]
    fn the_j2_bridge_regresses_its_node_at_the_secular_rate_in_both_precisions() {
        let (a, inc) = (7_000e3, 51.6_f64.to_radians());
        let j2 = 1.082_626_68e-3;
        let n = (MU_EARTH / (a * a * a)).sqrt();
        let speed = (MU_EARTH / a).sqrt();
        let r0 = [a, 0.0, 0.0];
        let v0 = [0.0, speed * inc.cos(), speed * inc.sin()];
        let (dt, steps) = (10.0, 8_640); // one day, ~583 steps per revolution
        // Kozai's first-order secular rate for a circular orbit: -3/2 n J2 (Re/a)^2 cos i.
        let raan_rate = -1.5 * n * j2 * (RE_EARTH / a).powi(2) * inc.cos();
        let raan = |r: [f64; 3], v: [f64; 3]| {
            let h = vec3(r).cross(vec3(v));
            h.x.atan2(-h.y)
        };

        Python::attach(|py| {
            let mut round_trip = Vec::new();
            for precision in ["f64", "dd"] {
                let (r1, v1) = propagate_j2_precision(
                    py, r0, v0, MU_EARTH, j2, RE_EARTH, dt, steps, precision,
                )
                .expect("both precision tags are accepted");
                // Measured drift after one day: -0.078 723 rad against the secular -0.077 999 5 rad.
                // The 7.2e-4 rad gap is the short-period term the closed form averages away, and it
                // is the size first-order theory predicts for one -- O(J2 cos i) = 6.7e-4. A wrong
                // sign or a dropped cos i in the oblateness moves the drift by a whole radian.
                let drift = raan(r1, v1) - raan(r0, v0);
                assert!(
                    (drift - raan_rate * dt * steps as f64).abs() < 1e-3,
                    "{precision}: node moved {drift} rad, secular theory says {}",
                    raan_rate * dt * steps as f64
                );

                let (r2, _) = propagate_j2_precision(
                    py, r1, v1, MU_EARTH, j2, RE_EARTH, -dt, steps, precision,
                )
                .expect("the integrator runs backwards as readily as forwards");
                round_trip.push(
                    (0..3)
                        .map(|i| (r2[i] - r0[i]).abs())
                        .fold(0.0_f64, f64::max),
                );
            }
            // Measured: 5.9e-5 m in f64 against 2.4e-8 m in double-double, a factor of 2440. The
            // double-double floor is the f64 the wrapper hands back, not the arithmetic inside it.
            assert!(
                round_trip[1] * 10.0 < round_trip[0],
                "the 'dd' arm must be the double-double one: round trip {:?}",
                round_trip
            );
            assert!(
                propagate_j2_precision(py, r0, v0, MU_EARTH, j2, RE_EARTH, dt, steps, "quad")
                    .is_err(),
                "an unknown precision tag must be rejected, not silently defaulted"
            );
        });
    }

    /// The Gauss-Newton bridge on a problem whose least-squares answer is arithmetic: six rows, each
    /// a unit partial on one component, so the normal equations are diagonal, the solution is the
    /// target the model was written around and the covariance is the inverse of the weights. This is
    /// the only path in the crate that calls *back* into Python mid-solve, so it is also where the
    /// callback's two failure modes are checked -- an exception the model raises has to reach the
    /// caller, and a model that cannot determine six states has to say so rather than hand back the
    /// initial guess.
    #[test]
    fn the_gauss_newton_bridge_solves_the_diagonal_system_its_model_describes() {
        let target = [1.5, -2.25, 0.75, 3.0, -0.5, 10.0];
        let sigma = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0];
        Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                c"
TARGET = [1.5, -2.25, 0.75, 3.0, -0.5, 10.0]
SIGMA = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0]


def diagonal(x):
    return [
        ([1.0 if j == i else 0.0 for j in range(6)], x[i] - TARGET[i], 1.0 / SIGMA[i] ** 2)
        for i in range(6)
    ]


def raising(x):
    raise KeyError('the model has no answer here')


def one_row(x):
    return [([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], x[0], 1.0)]
",
                c"residual_models.py",
                c"residual_models",
            )
            .expect("the fixture module compiles");

            let diagonal = module.getattr("diagonal").expect("diagonal is defined");
            let (x, cov, converged, iterations, _cost) =
                gauss_newton(&diagonal, [0.0; 6]).expect("a diagonal system of full rank solves");
            assert!(
                converged,
                "a linear system did not converge in {iterations}"
            );
            for i in 0..6 {
                assert!(
                    (x[i] - target[i]).abs() < 1e-9,
                    "component {i}: {} against {}",
                    x[i],
                    target[i]
                );
                assert!(
                    (cov[i][i] - sigma[i] * sigma[i]).abs() < 1e-9,
                    "variance {i}: {} against {}",
                    cov[i][i],
                    sigma[i] * sigma[i]
                );
            }

            let raising = module.getattr("raising").expect("raising is defined");
            assert!(
                gauss_newton(&raising, [0.0; 6]).is_err(),
                "the model's own exception has to reach the caller"
            );
            let one_row = module.getattr("one_row").expect("one_row is defined");
            assert!(
                gauss_newton(&one_row, [0.0; 6]).is_err(),
                "one row cannot determine six states"
            );
        });
    }

    /// The relativistic bundle against the same four forces integrated with the Sun's velocity from
    /// an independent five-point stencil at a ten-times-coarser step. The bridge derives it by a
    /// three-point central difference at 60 s; the two stencils agree to ~2e-11 relative and the
    /// de Sitter term they feed is ~4e-11 m/s^2, so the reference is exact for this purpose, while
    /// any slip in the difference -- a dropped sign, a denominator that is not 2h -- changes that
    /// term by orders of magnitude.
    #[test]
    fn the_relativistic_bridge_matches_an_independent_sun_velocity_stencil() {
        let r0 = [7_000e3, 0.0, 0.0];
        let v0 = [0.0, 6.2e3, 3.1e3];
        let (t1, epoch, mu_sun, tol) = (86_400.0, 7.2e8, 1.327_124_400_18e20, 1e-11);
        let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
        let controls = Controls::new(Tolerance::new(tol, tol), 1e-3);

        let sun_pv = move |t: f64| {
            let sun = AnalyticSun {
                epoch_j2000_seconds: epoch,
            };
            let h = 600.0;
            let (a, b, c, d) = (
                sun.position(t - 2.0 * h),
                sun.position(t - h),
                sun.position(t + h),
                sun.position(t + 2.0 * h),
            );
            let five = |i: usize| (a[i] - 8.0 * b[i] + 8.0 * c[i] - d[i]) / (12.0 * h);
            (sun.position(t), [five(0), five(1), five(2)])
        };
        let sys = Plain(Newtonian((
            PointMass { mu: MU_EARTH },
            Schwarzschild { mu: MU_EARTH },
            LenseThirring {
                mu: MU_EARTH,
                j: [0.0, 0.0, EARTH_ANGULAR_MOMENTUM_PER_MASS],
            },
            DeSitter { mu_sun, sun_pv },
        )));
        let reference = integrate_dop853(&sys, 0.0, &y0, t1, 86.4, controls)
            .expect("the reference system integrates over a day");
        let yr = reference.final_state();

        // How much of the answer is the de Sitter term at all: without it the same integration is
        // the number every mutation that kills the Sun's velocity converges to.
        let no_desitter = Plain(Newtonian((
            PointMass { mu: MU_EARTH },
            Schwarzschild { mu: MU_EARTH },
            LenseThirring {
                mu: MU_EARTH,
                j: [0.0, 0.0, EARTH_ANGULAR_MOMENTUM_PER_MASS],
            },
        )));
        let without = integrate_dop853(&no_desitter, 0.0, &y0, t1, 86.4, controls)
            .expect("the truncated system integrates too");
        let yw = without.final_state();

        Python::attach(|py| {
            let dense = propagate_relativistic_dense(py, r0, v0, t1, epoch, MU_EARTH, mu_sun, tol)
                .expect("the bridge integrates over a day");
            let (r, v) = dense.final_state();
            for i in 0..3 {
                // Measured over the day: the two stencils put the endpoint 9.1e-6 m apart, and the
                // de Sitter term is worth 5.1e-3 m of it. The bound sits an order of magnitude above
                // the first and two below the second, so it separates them either way -- a stencil
                // that returns zero lands on `yw`, and one that returns nonsense lands nowhere near.
                assert!(
                    (r[i] - yr[i]).abs() < 1e-4 && (v[i] - yr[i + 3]).abs() < 1e-7,
                    "component {i}: bridge {:?} against the five-point reference {yr:?}",
                    (r, v)
                );
                assert!(
                    (yr[i] - yw[i]).abs() > 1e-3,
                    "component {i}: de Sitter is worth {:e} m here, too little to test with",
                    yr[i] - yw[i]
                );
            }
        });
    }

    /// The EOP contract of the ITRF-frame J2 propagator: the time column needs at least two nodes to
    /// interpolate between, and every other column has to be as long as it. Four arities pin all
    /// five branches of that guard -- the comparison, its bound, the length equality, and the
    /// disjunction joining them -- because each one is the single case that tells the mutant apart:
    /// one node rejects, two accept, three accept, and a ragged three rejects.
    #[test]
    fn the_itrf_propagator_rejects_an_eop_series_it_cannot_interpolate() {
        let (r0, v0) = ([7_000e3, 0.0, 0.0], [0.0, 7.546e3, 0.0]);
        let run = |py: Python<'_>, t: Vec<f64>, xp: Vec<f64>| {
            let same = vec![0.0; t.len()];
            propagate_j2_itrf_eop_dense(
                py,
                r0,
                v0,
                600.0,
                MU_EARTH,
                1.082_626_68e-3,
                6_378_136.3,
                0,
                69.184,
                t,
                xp,
                same.clone(),
                same.clone(),
                same.clone(),
                same,
                1e-9,
            )
            .map(|_| ())
        };
        Python::attach(|py| {
            assert!(
                run(py, vec![0.0], vec![0.0]).is_err(),
                "a one-node series has no interval to interpolate over"
            );
            assert!(
                run(py, vec![0.0, 600.0], vec![0.0, 0.0]).is_ok(),
                "two nodes are the smallest series that does"
            );
            assert!(
                run(py, vec![0.0, 300.0, 600.0], vec![0.0; 3]).is_ok(),
                "three equal-length columns are a valid series"
            );
            assert!(
                run(py, vec![0.0, 300.0, 600.0], vec![0.0; 2]).is_err(),
                "a column shorter than the time column would index out of bounds"
            );
        });
    }

    /// Both dense ephemerides are built through an interpreter-facing entry point and then queried
    /// through plain accessors, so this pins the two halves together. The interpolant is checked
    /// *between* accepted steps -- which is the whole point of dense output -- against the
    /// universal-variable Kepler solution, a different algorithm rather than a replay of the same
    /// integration; `final_state` has to be `eval` at the right end of the span; and the step counts
    /// have to describe a real adaptive run rather than a constant.
    #[test]
    fn the_dense_ephemerides_interpolate_the_orbit_they_integrated() {
        let (r0, v0) = ([7_000e3, 0.0, 0.0], [0.0, 6.2e3, 3.1e3]);
        let (re, t1) = (6_378_137.0, 5_400.0);
        Python::attach(|py| {
            let dop = propagate_dense(py, r0, v0, t1, MU_EARTH, 0.0, re, 1e-12)
                .expect("a two-body arc integrates");
            let ias = propagate_ias15_dense(py, r0, v0, t1, MU_EARTH, 0.0, re)
                .expect("a two-body arc integrates");

            assert_eq!(
                dop.span(),
                (0.0, t1),
                "the DOP853 span is not what was asked"
            );
            assert_eq!(
                ias.span(),
                (0.0, t1),
                "the IAS15 span is not what was asked"
            );
            assert_eq!(dop.final_state(), dop.eval(t1), "DOP853 end != eval(t1)");
            assert_eq!(ias.final_state(), ias.eval(t1), "IAS15 end != eval(t1)");

            // Sample points chosen off any round fraction of the span, so they land inside accepted
            // steps rather than on their boundaries.
            for k in 1..=7 {
                let t = t1 * f64::from(k) / 7.3;
                let (rk, vk) = kepler(vec3(r0), vec3(v0), MU_EARTH, t);
                let (rk, vk) = (arr(rk), arr(vk));
                for (name, (r, v)) in [("dop853", dop.eval(t)), ("ias15", ias.eval(t))] {
                    for i in 0..3 {
                        assert!(
                            (r[i] - rk[i]).abs() < 1e-3 && (v[i] - vk[i]).abs() < 1e-6,
                            "{name} at t = {t} axis {i}: {:?} against Kepler {:?}",
                            (r[i], v[i]),
                            (rk[i], vk[i])
                        );
                    }
                }
            }

            for (name, (accepted, rejected)) in
                [("dop853", dop.step_counts()), ("ias15", ias.step_counts())]
            {
                assert!(accepted > 4, "{name} took {accepted} accepted steps");
                assert!(
                    rejected < accepted,
                    "{name} rejected {rejected} of {accepted}"
                );
            }
            // An orbit is not a straight line, so the interpolant cannot be its own end state.
            let middle = dop.eval(t1 / 2.0).0;
            assert!(
                (middle[0] - dop.final_state().0[0]).abs() > 1e5,
                "the DOP853 interpolant is constant across the span"
            );
        });
    }

    /// The RPC00B camera on the real Pleiades fixture. `forward` is pinned against the pixel `rpcm`
    /// produces for the same `.RPB` and the same ground point -- an oracle outside this repository --
    /// and `inverse` against its ground point; then the two are required to undo each other, which
    /// no pair of constants can do. The block adjustment is given control points carrying a shift
    /// this test built, and has to hand that shift back at a point it never saw.
    #[test]
    fn the_rpc_bindings_reproduce_the_rpcm_reference_and_invert_each_other() {
        const RPB: &str =
            include_str!("../../spacelab-earthvision/tests/fixtures/pleiades_reunion.rpb");
        Python::attach(|_| {
            let rpc = PyRpc::parse_rpb(RPB).expect("the fixture is a valid RPB");
            let (row, col) = rpc.forward(-21.231_608_128_8, 55.711_969_880_1, 1295.0);
            assert!(
                (row - 313.646_096_127_999_04).abs() < 1e-4
                    && (col - 13_058.594_417_715_2).abs() < 1e-4,
                "forward gave ({row}, {col})"
            );
            let (lat, lon) = rpc
                .inverse(5000.0, 8000.0, 1595.0)
                .expect("a pixel inside the fitted image has a ground point");
            assert!(
                (lat + 21.252_398_282_141_336).abs() < 1e-6
                    && (lon - 55.687_077_361_188_35).abs() < 1e-6,
                "inverse gave ({lat}, {lon})"
            );
            let (r2, c2) = rpc.forward(lat, lon, 1595.0);
            assert!(
                (r2 - 5000.0).abs() < 1e-6 && (c2 - 8000.0).abs() < 1e-6,
                "the round trip landed at ({r2}, {c2})"
            );
            assert!(
                PyRpc::parse_rpb("LINE_OFF: 0;").is_err(),
                "an incomplete RPB must be rejected"
            );

            // Control points carrying a constant image shift; a shift-only adjustment recovers it.
            let (d_row, d_col) = (12.0, -7.0);
            let gcps: Vec<(f64, f64, f64, f64, f64)> = [
                (-21.24, 55.70, 1200.0),
                (-21.22, 55.72, 1400.0),
                (-21.26, 55.74, 1300.0),
                (-21.20, 55.68, 1100.0),
            ]
            .into_iter()
            .map(|(la, lo, h)| {
                let (r, c) = rpc.forward(la, lo, h);
                (la, lo, h, r + d_row, c + d_col)
            })
            .collect();
            let query = (-21.23, 55.71, 1300.0);
            let (r_adj, c_adj) = rpc_block_adjust(&rpc, gcps, query, false);
            let (r0, c0) = rpc.forward(query.0, query.1, query.2);
            assert!(
                (r_adj - (r0 + d_row)).abs() < 1e-6 && (c_adj - (c0 + d_col)).abs() < 1e-6,
                "the adjustment gave ({r_adj}, {c_adj}) against ({}, {})",
                r0 + d_row,
                c0 + d_col
            );
        });
    }

    /// The POD reduced-order density model on snapshots that lie exactly in a two-dimensional
    /// subspace: a rank-2 basis then captures all of the variance, projection followed by
    /// reconstruction is the identity on the training fields, and the mean field projects to the
    /// origin. Every one of those is a constructed truth rather than a stored table.
    #[test]
    fn the_density_rom_reconstructs_the_subspace_it_was_fit_on() {
        let n = 12_usize;
        let mode_a: Vec<f64> = (0..n).map(|k| ((k as f64) * 0.7).sin()).collect();
        let mode_b: Vec<f64> = (0..n).map(|k| ((k as f64) * 0.3).cos()).collect();
        let base: Vec<f64> = (0..n).map(|k| 1.0 + 0.05 * k as f64).collect();
        let weights = [
            (-2.0, 1.0),
            (-1.0, -0.5),
            (0.5, 2.0),
            (1.5, -1.5),
            (3.0, 0.25),
        ];
        let snapshots: Vec<Vec<f64>> = weights
            .iter()
            .map(|&(a, b)| {
                (0..n)
                    .map(|k| base[k] + a * mode_a[k] + b * mode_b[k])
                    .collect()
            })
            .collect();

        Python::attach(|_| {
            let rom = PyDensityRom::fit(snapshots.clone(), 2).expect("five snapshots span rank 2");
            assert_eq!(rom.rank(), 2, "the retained rank is not what was asked");
            assert_eq!(rom.dimension(), n, "the field dimension changed");
            assert!(
                (rom.energy_fraction() - 1.0).abs() < 1e-9,
                "a rank-2 subspace captured only {}",
                rom.energy_fraction()
            );
            // Truncating to one mode has to leave variance behind, and exactly the share the
            // retained coordinate accounts for. The oracle is built from `project` and `mean`
            // rather than from the fraction itself: retained energy over total centred energy.
            let truncated = PyDensityRom::fit(snapshots.clone(), 1).expect("rank 1 is admissible");
            let centre = truncated.mean();
            let (mut retained, mut total) = (0.0_f64, 0.0_f64);
            for s in &snapshots {
                retained += truncated
                    .project(s.clone())
                    .iter()
                    .map(|c| c * c)
                    .sum::<f64>();
                total += (0..n).map(|k| (s[k] - centre[k]).powi(2)).sum::<f64>();
            }
            assert!(
                (truncated.energy_fraction() - retained / total).abs() < 1e-9,
                "rank-1 energy {} against the projected share {}",
                truncated.energy_fraction(),
                retained / total
            );
            assert!(
                truncated.energy_fraction() < 0.999,
                "one mode of a genuinely two-dimensional subspace cannot capture {}",
                truncated.energy_fraction()
            );

            let mean = rom.mean();
            assert_eq!(mean.len(), n, "the mean field changed length");
            for (k, &m) in mean.iter().enumerate() {
                let want = snapshots.iter().map(|s| s[k]).sum::<f64>() / snapshots.len() as f64;
                assert!((m - want).abs() < 1e-12, "mean {k}: {m} against {want}");
            }
            for c in rom.project(mean.clone()) {
                assert!(c.abs() < 1e-9, "the mean field projects to {c}, not 0");
            }

            for (i, field) in snapshots.iter().enumerate() {
                let coords = rom.project(field.clone());
                assert_eq!(
                    coords.len(),
                    2,
                    "snapshot {i} got {} coordinates",
                    coords.len()
                );
                let back = rom.reconstruct(coords);
                assert_eq!(back.len(), n);
                for (k, (&g, &w)) in back.iter().zip(field).enumerate() {
                    assert!(
                        (g - w).abs() < 1e-9,
                        "snapshot {i} pixel {k}: {g} against {w}"
                    );
                }
            }

            assert!(
                PyDensityRom::fit(vec![vec![1.0, 2.0]], 1).is_err(),
                "one snapshot cannot define a basis"
            );
        });
    }

    /// The two learned dynamics layers, each fit on data generated from a linear map this test
    /// chose: the static driver→coordinate map has to reproduce `W` on a driver row it never saw,
    /// and DMDc has to hand `A` and `B` back and step the state forward with them.
    #[test]
    fn the_learned_dynamics_return_the_linear_maps_they_were_fit_on() {
        // alpha = W·d, W is 2x3.
        let w = [[0.5, -1.25, 2.0], [-3.0, 0.75, 0.125]];
        let driver_rows: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 2.0, -1.0],
            vec![-2.0, 0.5, 3.0],
        ];
        let apply = |m: &[[f64; 3]; 2], d: &[f64]| -> Vec<f64> {
            m.iter()
                .map(|row| row.iter().zip(d).map(|(a, b)| a * b).sum())
                .collect()
        };
        let coord_rows: Vec<Vec<f64>> = driver_rows.iter().map(|d| apply(&w, d)).collect();

        // a_{k+1} = A·a_k + B·u_k, A 2x2 and B 2x2.
        let (a, b) = ([[0.9, 0.1], [-0.2, 0.8]], [[1.5, 0.0], [0.0, -2.5]]);
        let prev: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![-1.0, 2.0],
            vec![3.0, -0.5],
        ];
        let controls: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 1.0],
            vec![2.0, 0.5],
            vec![0.25, -3.0],
        ];
        let step = |s: &[f64], u: &[f64]| -> Vec<f64> {
            (0..2)
                .map(|i| a[i][0] * s[0] + a[i][1] * s[1] + b[i][0] * u[0] + b[i][1] * u[1])
                .collect()
        };
        let next: Vec<Vec<f64>> = prev
            .iter()
            .zip(&controls)
            .map(|(s, u)| step(s, u))
            .collect();

        Python::attach(|_| {
            let dynamics = PyReducedDynamics::fit(driver_rows, coord_rows)
                .expect("five exact rows determine a 2x3 map");
            let fresh = vec![0.3, -1.7, 4.25];
            let predicted = dynamics.predict(fresh.clone());
            let want = apply(&w, &fresh);
            assert_eq!(predicted.len(), 2, "the prediction is two coordinates");
            for (k, (&g, &t)) in predicted.iter().zip(&want).enumerate() {
                assert!((g - t).abs() < 1e-9, "coordinate {k}: {g} against {t}");
            }

            let dmdc = PyDmdc::fit(prev, controls, next).expect("five exact steps determine [A|B]");
            let got_a = dmdc.system_matrix();
            let got_b = dmdc.control_matrix();
            assert_eq!((got_a.len(), got_b.len()), (2, 2), "the operators are 2x2");
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (got_a[i][j] - a[i][j]).abs() < 1e-9,
                        "A[{i}][{j}] = {} against {}",
                        got_a[i][j],
                        a[i][j]
                    );
                    assert!(
                        (got_b[i][j] - b[i][j]).abs() < 1e-9,
                        "B[{i}][{j}] = {} against {}",
                        got_b[i][j],
                        b[i][j]
                    );
                }
            }
            let (state, control) = (vec![0.7, -1.2], vec![2.0, 0.4]);
            let stepped = dmdc.predict_next(state.clone(), control.clone());
            let truth = step(&state, &control);
            assert_eq!(
                stepped.len(),
                truth.len(),
                "a short step vector would zip to nothing and assert nothing"
            );
            for (k, (&g, &t)) in stepped.iter().zip(&truth).enumerate() {
                assert!((g - t).abs() < 1e-9, "state {k}: {g} against {t}");
            }
        });
    }

    /// The square-root information filter on a problem whose answer is arithmetic: six independent
    /// unit-partial rows make the estimate the observations themselves, so the solve, the residual
    /// sum and the row count are all known in advance. A seventh, inconsistent row then has to move
    /// the residual sum off zero -- which a constant pair cannot follow.
    #[test]
    fn the_srif_counts_its_rows_and_reports_the_residual_it_carries() {
        Python::attach(|_| {
            let truth = [10.0, -20.0, 30.0, -0.5, 0.25, -0.125];
            let mut srif = Srif6::new();
            assert_eq!(
                srif.residual_stats(),
                (0.0, 0),
                "an empty filter has no rows"
            );
            for (i, &z) in truth.iter().enumerate() {
                let mut h = [0.0; 6];
                h[i] = 1.0;
                srif.add_row(h, z, 4.0);
            }
            let (sum, rows) = srif.residual_stats();
            assert_eq!(rows, 6, "six rows went in");
            assert!(sum.abs() < 1e-18, "a consistent system left residual {sum}");

            let solved = srif.solve().expect("six independent rows are full rank");
            for (i, (&g, &t)) in solved.iter().zip(&truth).enumerate() {
                assert!((g - t).abs() < 1e-9, "state {i}: {g} against {t}");
            }

            // A second, contradictory measurement of x0: weighted least squares splits the
            // difference and the weighted residual sum stops being zero.
            let mut h = [0.0; 6];
            h[0] = 1.0;
            srif.add_row(h, truth[0] + 2.0, 4.0);
            let (sum, rows) = srif.residual_stats();
            assert_eq!(rows, 7, "the seventh row was not counted");
            assert!(sum > 1.0, "an inconsistent system left residual {sum}");
            assert!(
                Srif6::new().solve().is_err(),
                "a filter with no rows is rank-deficient"
            );
        });
    }

    /// The density-calibration pair: a correction fitted to samples drawn from a known factor field
    /// has to give those 25 coefficients back through `density_correction_factor`, and the grid
    /// snapshot has to bin the same samples into the shape it was asked for. Both report the
    /// length mismatch their doc comments promise rather than indexing past the end.
    #[test]
    fn the_density_calibration_bindings_recover_the_field_they_were_given() {
        // A smooth sun-fixed field, sampled on a lat x local-time grid.
        let (n_lat, n_lst) = (6_usize, 8_usize);
        let (mut lats, mut lsts, mut ratios, mut weights) = (vec![], vec![], vec![], vec![]);
        for i in 0..n_lat {
            for j in 0..n_lst {
                let lat = -1.4 + 2.8 * (i as f64 + 0.5) / n_lat as f64;
                let lst = 24.0 * (j as f64 + 0.5) / n_lst as f64;
                lats.push(lat);
                lsts.push(lst);
                ratios.push(0.2 * lat.sin() + 0.1 * (core::f64::consts::PI * lst / 12.0).cos());
                weights.push(1.0);
            }
        }

        Python::attach(|_| {
            let coefficients =
                density_correction_fit(lats.clone(), lsts.clone(), ratios.clone(), weights, 1e-6)
                    .expect("48 samples determine 25 degree-4 coefficients");
            assert_eq!(coefficients.len(), 25, "a degree-4 correction is 25 terms");
            // The fitted factor is exp(ln-ratio), so it has to track the field it was fitted to.
            for k in [0_usize, 17, 31, 47] {
                let factor = density_correction_factor(coefficients.clone(), lats[k], lsts[k])
                    .expect("25 coefficients is the documented length");
                assert!(
                    (factor.ln() - ratios[k]).abs() < 0.05,
                    "sample {k}: ln(factor) {} against {}",
                    factor.ln(),
                    ratios[k]
                );
            }
            assert!(
                density_correction_factor(vec![0.0; 24], 0.0, 12.0).is_err(),
                "24 coefficients is not a degree-4 correction"
            );
            // Each column short in turn: a guard that only reads the first one indexes past the end
            // of the others, so the three cases have to be separate calls, not one ragged shape.
            for short in 1..4_usize {
                let mut lengths = [3_usize; 4];
                lengths[short] = 2;
                assert!(
                    density_correction_fit(
                        vec![0.0; lengths[0]],
                        vec![0.0; lengths[1]],
                        vec![0.0; lengths[2]],
                        vec![1.0; lengths[3]],
                        0.0,
                    )
                    .is_err(),
                    "column {short} short: ragged input must be rejected, not indexed past the end"
                );
            }

            let grid =
                density_grid_snapshot(lats.clone(), lsts.clone(), ratios.clone(), n_lat, n_lst)
                    .expect("the three columns are aligned");
            assert_eq!(grid.len(), n_lat * n_lst, "the snapshot is lat x lst");
            // One sample per cell, so each cell is that sample's own log-ratio.
            for (k, &r) in ratios.iter().enumerate() {
                assert!(
                    (grid[k] - r).abs() < 1e-12,
                    "cell {k}: {} against {r}",
                    grid[k]
                );
            }
            for short in 1..3_usize {
                let mut lengths = [3_usize; 3];
                lengths[short] = 2;
                assert!(
                    density_grid_snapshot(
                        vec![0.0; lengths[0]],
                        vec![0.0; lengths[1]],
                        vec![0.0; lengths[2]],
                        n_lat,
                        n_lst,
                    )
                    .is_err(),
                    "column {short} short: ragged input must be rejected"
                );
            }
        });
    }

    /// The four DSST entry points on one equinoctial state. Three properties hold them down without
    /// a stored trajectory: a zero-length step returns the state it was given; the mean semi-major
    /// axis is conserved under the zonal field, which is the defining property of mean elements;
    /// and the osculating output differs from the mean one by the short-period corrections it adds.
    /// The `_full` pair then has to differ from the zonal-only pair, because it carries the third
    /// bodies and solar radiation pressure the zonal one does not.
    #[test]
    fn the_dsst_bindings_conserve_the_mean_axis_and_add_the_short_period_terms() {
        // [a, k, h, q, p, λ] -- a slightly eccentric, inclined low orbit.
        let initial = [7_200e3, 1.0e-3, 5.0e-4, 0.2, 0.1, 0.3];
        let times = vec![0.0, 5_400.0, 10_800.0];
        let (mu, re) = (3.986_004_415e14, 6_378_136.3);
        let j = std::vec![0.0, 0.0, 1.082_626_68e-3, -2.532_653_65e-6];

        Python::attach(|py| {
            let mean = dsst_zonal_propagate_mean(py, initial, times.clone(), mu, re, j.clone());
            let osculating =
                dsst_zonal_propagate_osculating(py, initial, times.clone(), mu, re, j.clone());
            assert_eq!(mean.len(), times.len(), "one mean state per epoch");
            assert_eq!(
                osculating.len(),
                times.len(),
                "one osculating state per epoch"
            );
            assert_eq!(mean[0], initial, "a zero-length step moved the mean state");

            for (i, state) in mean.iter().enumerate() {
                assert!(
                    (state[0] - initial[0]).abs() < 1e-6,
                    "the mean axis drifted to {} at epoch {i}",
                    state[0]
                );
            }
            // The mean longitude runs at the Keplerian mean motion, to within the J2 correction.
            let n = (mu / initial[0].powi(3)).sqrt();
            let advance = mean[2][5] - mean[0][5];
            assert!(
                (advance / (n * times[2]) - 1.0).abs() < 0.01,
                "λ advanced {advance} rad against n·t = {}",
                n * times[2]
            );

            // Short-period corrections are the difference between the two outputs, and they are
            // kilometres in `a` -- small against the orbit, far above any rounding.
            let short_period = (osculating[1][0] - mean[1][0]).abs();
            assert!(
                (1.0..50e3).contains(&short_period),
                "the short-period correction in a is {short_period} m"
            );

            let (cr, area_over_mass) = (1.5, 0.02);
            let full_mean = dsst_propagate_mean_full(
                py,
                initial,
                times.clone(),
                0.0,
                mu,
                re,
                j.clone(),
                cr,
                area_over_mass,
            );
            let full_osculating = dsst_propagate_osculating_full(
                py,
                initial,
                times.clone(),
                0.0,
                mu,
                re,
                j,
                cr,
                area_over_mass,
            );
            assert_eq!(full_mean.len(), times.len());
            assert_eq!(full_osculating.len(), times.len());
            assert_eq!(
                full_mean[0], initial,
                "a zero-length full step moved the state"
            );
            // Sun, Moon and SRP move the mean longitude off the zonal-only answer.
            assert!(
                (full_mean[2][5] - mean[2][5]).abs() > 1e-12,
                "the full model reproduced the zonal-only mean exactly"
            );
            assert!(
                (full_osculating[1][0] - full_mean[1][0]).abs() > 1.0,
                "the full osculating output carries no short-period correction"
            );
        });
    }

    /// The pole-tide binding's `mean_pole` argument is a two-valued switch, and the two conventional
    /// poles have drifted far enough apart that the same polar motion gives wobble variables of
    /// wholly different size through them. Measured at 24 years past J2000 and 0.2 arcsec of polar
    /// motion, ΔC21 is -1.494e-10 through the 2018 secular pole against -5.453e-13 through the
    /// retired 2010 one -- 274x, which no defaulted switch reproduces. Anything but those two names
    /// is rejected rather than silently taking one of them.
    #[test]
    fn the_pole_tide_binding_distinguishes_the_two_conventional_poles() {
        let (xp, yp, t) = (1.0e-6, 1.5e-6, 757_382_400.0);
        Python::attach(|_| {
            let modern = solid_pole_tide(xp, yp, t, "secular2018").expect("the default pole");
            let retired = solid_pole_tide(xp, yp, t, "iers2010").expect("the Orekit pole");
            assert!(
                (modern[0] / retired[0]).abs() > 10.0,
                "the two poles gave ΔC21 = {} and {}",
                modern[0],
                retired[0]
            );
            assert!(
                solid_pole_tide(xp, yp, t, "mean").is_err(),
                "an unknown pole must be rejected, not defaulted"
            );
        });
    }
    /// `epoch_utc_ns`, `tt_minus_utc_s` and the `eop_dut1` column are three ways of writing the same
    /// two timelines: the propagator can only see UTC + (TT-UTC) and UTC + dUT1, so moving the epoch
    /// forward by a day and taking a day off both offsets has to leave the trajectory where it was.
    /// Every arithmetic slip in assembling those epochs breaks that -- a `-` for a `+` doubles the
    /// shift instead of cancelling it, a `*` for a `+` scales it, and a `/ 1e9` for a `* 1e9` freezes
    /// the clock -- and the compensated pair then lands a day apart on a timeline whose reference
    /// value is exactly zero. Measured: 0 m as written, against 6.4e-5 m for one day of UT1 error and
    /// 1.3e-3 m for two days of TT error.
    ///
    /// The polar motion is not decoration. A zonal J2 is invariant under rotation about the pole, so
    /// with `xp = yp = 0` the ITRF z-axis *is* the CIP and UT1 cancels exactly -- 4.7e-9 m for a
    /// 100 s error, which is f64 noise in the rotation and nothing else. At a realistic 0.2" the
    /// pole and the rotation axis part company and the same error is worth 2.7e-5 m.
    #[test]
    fn the_itrf_propagator_reads_its_epoch_and_its_offsets_as_one_timeline() {
        let (r0, v0) = ([7_000e3, 0.0, 0.0], [0.0, 7.546e3, 0.0]);
        let epoch: i128 = 631_152_069_184_000_000; // 2020-01-01T00:00:00 UTC, ns since J2000
        let (day_ns, day_s) = (86_400_000_000_000_i128, 86_400.0);
        let run = |py: Python<'_>, epoch: i128, tt_minus_utc: f64, dut1: f64| {
            let pm = 1e-6; // ~0.2", the scale polar motion actually runs at
            propagate_j2_itrf_eop_dense(
                py,
                r0,
                v0,
                600.0,
                MU_EARTH,
                1.082_626_68e-3,
                6_378_136.3,
                epoch,
                tt_minus_utc,
                vec![0.0, 600.0],
                vec![pm, pm],
                vec![pm, pm],
                vec![dut1, dut1],
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                1e-9,
            )
            .map(|e| e.final_state().0)
            .expect("a two-node EOP series over the propagated span is valid")
        };
        Python::attach(|py| {
            let here = run(py, epoch, 69.184, 0.3);
            let day_later = run(py, epoch + day_ns, 69.184 - day_s, 0.3 - day_s);
            for i in 0..3 {
                assert!(
                    (here[i] - day_later[i]).abs() < 1e-9,
                    "component {i}: {here:?} against the compensated {day_later:?}"
                );
            }
        });
    }

    /// The drag propagator's inertial-to-geodetic closure, against the same rotation applied through
    /// the typed frame API instead of by multiplying the three rows out by hand. Altitude is what
    /// the atmosphere reads, and a garbled rotation moves it by thousands of kilometres, so the
    /// density it feeds the drag differs by orders of magnitude — nothing subtle survives here.
    #[test]
    fn the_drag_propagator_puts_the_satellite_where_the_typed_frame_transform_does() {
        let (r0, v0) = ([6_778e3, 0.0, 0.0], [0.0, 5.4e3, 5.4e3]);
        let t1 = 3_600.0;
        let (epoch_tt, epoch_ut1) = (631_152_069_184_000_000_i128, 631_152_000_000_000_000_i128);
        let (doy, sec, bc) = (1, 0.0, 0.02); // C_d·A/m, m²/kg — a real satellite, not a sail
        let (f107, f107a, ap) = (150.0, 150.0, 4.0);
        let (xp, yp, dx, dy, tol) = (1e-6, 1e-6, 0.0, 0.0, 1e-10);

        let cio = cio_fast(epoch_tt, epoch_ut1, t1, xp, yp, dx, dy);
        let geodetic = move |t: f64, rf: [f64; 3]| match earth_itrf_to_geodetic(
            cio.transform(t).position(Position::<Gcrf>::new(vec3(rf))),
        ) {
            Ok(g) => (
                g.height.metres() / 1000.0,
                g.latitude.degrees(),
                g.longitude.degrees(),
            ),
            Err(_) => (0.0, 0.0, 0.0),
        };
        let sys = Plain(Newtonian((
            PointMass { mu: MU_EARTH },
            Drag {
                ballistic_coefficient: bc,
                earth_rotation_rate: EARTH_ROTATION_RATE,
                atmosphere: Nrlmsise00Atmosphere {
                    doy0: doy,
                    sec0: sec,
                    f107,
                    f107a,
                    ap,
                    geodetic,
                },
                wind: NoWind,
            },
        )));
        let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
        let reference = integrate_dop853(
            &sys,
            0.0,
            &y0,
            t1,
            (t1 / 1000.0).abs().max(1.0),
            Controls::new(Tolerance::new(tol, tol), 1e-3),
        )
        .expect("the reference system integrates over an hour")
        .final_state();

        Python::attach(|py| {
            let dense = propagate_drag_nrlmsise00_dense(
                py, r0, v0, t1, epoch_tt, epoch_ut1, doy, sec, bc, f107, f107a, ap, MU_EARTH, xp,
                yp, dx, dy, 0.0, tol,
            )
            .expect("the bridge integrates over an hour");
            let (r, v) = dense.final_state();
            for i in 0..3 {
                assert!(
                    (r[i] - reference[i]).abs() < 1e-6 && (v[i] - reference[i + 3]).abs() < 1e-9,
                    "component {i}: bridge {:?} against the typed-transform reference {reference:?}",
                    (r, v)
                );
            }
            // The drag has to be doing something, or the comparison above is a two-body identity.
            let free = propagate_two_body(py, r0, v0, MU_EARTH, 1.0, 3_600, "f64")
                .expect("the two-body reference runs");
            let decayed: f64 = (0..3)
                .map(|i| (r[i] - free.0[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(decayed > 1.0, "drag moved the endpoint by only {decayed} m");
        });
    }

    /// The SDTM-1 drag propagator against the same system assembled here from the crate's own types,
    /// through the typed frame transform: the shipped field, the storm state run over the history, the
    /// two fluxes each in its slot, the epoch and the ballistic coefficient. The two fluxes are
    /// different numbers and the history ends in a storm, so exchanging the fluxes or quieting the
    /// history moves the density and so the endpoint.
    #[test]
    fn the_sdtm1_drag_propagator_is_the_crates_own_system() {
        let (r0, v0) = ([6_778e3, 0.0, 0.0], [0.0, 5.4e3, 5.4e3]);
        let t1 = 3_600.0;
        let (epoch_tt, epoch_ut1) = (631_152_069_184_000_000_i128, 631_152_000_000_000_000_i128);
        let (doy, sec, bc) = (172.0, 29_000.0, 0.02); // C_d·A/m, m²/kg
        let (f107, f107a) = (128.0, 141.0);
        let history: Vec<f64> = (0..120)
            .map(|i| if i < 104 { 7.0 } else { 120.0 })
            .collect();
        let storm = SDTM1_FIELD.storm().response(&history, 10_800.0);
        let (xp, yp, dx, dy, tol) = (1e-6, 1e-6, 0.0, 0.0, 1e-10);

        let cio = cio_fast(epoch_tt, epoch_ut1, t1, xp, yp, dx, dy);
        let geodetic = move |t: f64, rf: [f64; 3]| match earth_itrf_to_geodetic(
            cio.transform(t).position(Position::<Gcrf>::new(vec3(rf))),
        ) {
            Ok(g) => (
                g.height.metres() / 1000.0,
                g.latitude.degrees(),
                g.longitude.degrees(),
            ),
            Err(_) => (0.0, 0.0, 0.0),
        };
        let sys = Plain(Newtonian((
            PointMass { mu: MU_EARTH },
            Drag {
                ballistic_coefficient: bc,
                earth_rotation_rate: EARTH_ROTATION_RATE,
                atmosphere: SdtmAtmosphere {
                    field: SDTM1_FIELD,
                    drivers: SdtmDrivers { f107, f107a, storm },
                    doy0: doy,
                    sec0: sec,
                    geodetic,
                },
                wind: NoWind,
            },
        )));
        let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
        let reference = integrate_dop853(
            &sys,
            0.0,
            &y0,
            t1,
            (t1 / 1000.0).abs().max(1.0),
            Controls::new(Tolerance::new(tol, tol), 1e-3),
        )
        .expect("the reference system integrates over an hour")
        .final_state();

        Python::attach(|py| {
            let attempt = |f107: f64, f107a: f64, history: &[f64], cadence_s: f64| {
                propagate_drag_sdtm1_dense(
                    py,
                    r0,
                    v0,
                    t1,
                    epoch_tt,
                    epoch_ut1,
                    doy,
                    sec,
                    bc,
                    history.to_vec(),
                    f107,
                    f107a,
                    cadence_s,
                    MU_EARTH,
                    xp,
                    yp,
                    dx,
                    dy,
                    0.0,
                    tol,
                )
            };
            let run = |f107: f64, f107a: f64, history: &[f64]| {
                attempt(f107, f107a, history, 10_800.0)
                    .expect("the bridge integrates over an hour")
                    .final_state()
            };
            let (r, v) = run(f107, f107a, &history);
            for i in 0..3 {
                assert!(
                    (r[i] - reference[i]).abs() < 1e-6 && (v[i] - reference[i + 3]).abs() < 1e-9,
                    "component {i}: bridge {:?} against the crate's own system {reference:?}",
                    (r, v)
                );
            }
            // Each driver reaches the field: a swap of the two fluxes, or a quiet history, is another
            // orbit. A history of one value is that value's steady state, which is not the stormy one.
            let apart = |(s, _): ([f64; 3], [f64; 3])| {
                (0..3).map(|i| (s[i] - r[i]).powi(2)).sum::<f64>().sqrt()
            };
            assert!(apart(run(f107a, f107, &history)) > 1e-3);
            assert!(apart(run(f107, f107a, &[7.0; 120])) > 1e-3);
            assert!(apart(run(f107, f107a, &[7.0])) > 1e-3);
            // And a history that cannot be one is refused before anything is integrated, by name: an
            // error from the integration instead would come from a storm state the refusal let through.
            for (bad, cadence_s, named) in [
                (vec![], 10_800.0, "ap_history is empty"),
                (vec![7.0, f64::NAN, 7.0], 10_800.0, "ap_history[1] is NaN"),
                (vec![7.0, f64::INFINITY], 10_800.0, "ap_history[1] is inf"),
                (history.clone(), 0.0, "cadence_s is 0,"),
                (history.clone(), -10_800.0, "cadence_s is -10800,"),
                (history.clone(), f64::NAN, "cadence_s is NaN,"),
                (history.clone(), f64::INFINITY, "cadence_s is inf,"),
            ] {
                let Err(refused) = attempt(f107, f107a, &bad, cadence_s) else {
                    panic!("{bad:?} at {cadence_s} s accepted");
                };
                assert!(
                    refused.is_instance_of::<PyValueError>(py)
                        && refused.to_string().contains(named),
                    "{bad:?} at {cadence_s} s: {refused}"
                );
            }
        });
    }

    /// The tide propagator's Step-2 closure, against the same corrections assembled here from the
    /// same primitives: the two epochs advance with `t`, the pole-tide arguments are seconds of TT
    /// since J2000, and the solid and ocean pole tides *add* to `ΔC̄21` and `ΔS̄21` rather than
    /// replacing or subtracting them. The reference takes its seconds from `Epoch::day_and_fraction`
    /// instead of dividing the raw nanosecond count, which is the more accurate of the two routes
    /// and agrees with it to far better than this assertion.
    #[test]
    fn the_tide_propagator_assembles_step_two_on_the_timeline_it_documents() {
        let (r0, v0) = ([6_878e3, 0.0, 0.0], [0.0, 5.4e3, 5.4e3]);
        let (t1, degree) = (3_600.0, 8usize);
        let (epoch_tt, epoch_ut1) = (631_152_069_184_000_000_i128, 631_152_000_000_000_000_i128);
        let (epoch_j2000, mu_sun, mu_moon) =
            (631_152_069.184, 1.327_124_400_18e20, 4.902_800_066e12);
        let (xp, yp, tol) = (1e-6, 1e-6, 1e-10);

        let geo = GeoPotential::embedded(degree).expect("degree 8 is embedded");
        let cio = std::sync::Arc::new(cio_fast(epoch_tt, epoch_ut1, t1, xp, yp, 0.0, 0.0));
        let cio_tide = std::sync::Arc::clone(&cio);
        let tide = SolidEarthTide {
            mu_earth: 3.986_004_415e14,
            re_earth: 6_378_136.3,
            mu_sun,
            mu_moon,
            sun: AnalyticSun {
                epoch_j2000_seconds: epoch_j2000,
            },
            moon: AnalyticMoon {
                epoch_j2000_seconds: epoch_j2000,
            },
            rotate: move |t: f64| cio_tide.rotation_rows(t),
            step2: move |t: f64| {
                let dt_ns = (t * 1e9).round() as i128;
                let tt = Epoch::<Tt>::from_ns_since_j2000(epoch_tt + dt_ns);
                let ut1 = Epoch::<Ut1>::from_ns_since_j2000(epoch_ut1 + dt_ns);
                let tc = tt.julian_centuries();
                let mut s = spacelab_propagation::step2_corrections(
                    spacelab_fa::gmst82(ut1).radians(),
                    [
                        spacelab_fa::fundamental::fal03(tc),
                        spacelab_fa::fundamental::falp03(tc),
                        spacelab_fa::fundamental::faf03(tc),
                        spacelab_fa::fundamental::fad03(tc),
                        spacelab_fa::fundamental::faom03(tc),
                    ],
                );
                let (day, frac) = tt.day_and_fraction();
                let seconds = (day as f64) * 86_400.0 + frac * 86_400.0;
                let pole = spacelab_propagation::MeanPole::Secular2018;
                let sp = spacelab_propagation::pole_tide_corrections(xp, yp, seconds, pole);
                let op = spacelab_propagation::ocean_pole_tide_corrections(xp, yp, seconds, pole);
                s[1] += sp[0] + op[0];
                s[2] += sp[1] + op[1];
                s
            },
        };
        let sys = Plain(Newtonian((
            BodyOriented {
                inner: geo,
                rotate: move |t: f64| cio.rotation_rows(t),
            },
            tide,
        )));
        let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
        let reference = integrate_dop853(
            &sys,
            0.0,
            &y0,
            t1,
            (t1 / 1000.0).abs().max(1.0),
            Controls::new(Tolerance::new(tol, tol), 1e-3),
        )
        .expect("the reference system integrates over an hour")
        .final_state();

        Python::attach(|py| {
            let dense = propagate_geopotential_tide_dense(
                py,
                r0,
                v0,
                t1,
                degree,
                epoch_tt,
                epoch_ut1,
                epoch_j2000,
                xp,
                yp,
                0.0,
                0.0,
                0.0,
                mu_sun,
                mu_moon,
                tol,
            )
            .expect("the bridge integrates over an hour");
            let (r, v) = dense.final_state();
            for i in 0..3 {
                assert!(
                    (r[i] - reference[i]).abs() < 1e-6 && (v[i] - reference[i + 3]).abs() < 1e-9,
                    "component {i}: bridge {:?} against the reference {reference:?}",
                    (r, v)
                );
            }
        });
    }

    /// The ONNX cloud binding against onnxruntime's own probabilities for the same fixture, which is
    /// an oracle from outside this code path rather than a re-run of it. Only compiled with
    /// `--features onnx`, and until it is, every mutant of `cloud_mask` survives for the same reason
    /// the `extension-module` ones used to: the function is not in the build, so nothing can see it.
    #[cfg(feature = "onnx")]
    #[test]
    fn the_cloud_binding_thresholds_the_probabilities_onnxruntime_reports() {
        fn read_f32(bytes: &[u8]) -> Vec<f32> {
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        let onnx = include_bytes!("../../spacelab-vision/tests/fixtures/cloud_tiny.onnx").to_vec();
        let tile = read_f32(include_bytes!(
            "../../spacelab-vision/tests/fixtures/cloud_tiny_input.f32"
        ));
        let reference = read_f32(include_bytes!(
            "../../spacelab-vision/tests/fixtures/cloud_tiny_output.f32"
        ));
        // A 3x8x8 tile through a net that halves the grid: 16 sigmoid outputs, straddling 0.5.
        assert_eq!(reference.len(), 16);
        assert!(
            reference.iter().any(|&p| p >= 0.5) && reference.iter().any(|&p| p < 0.5),
            "the fixture must straddle the cut to say anything about a threshold"
        );

        for cut in [0.0, 0.5, 1.5] {
            let mask = cloud_mask(onnx.clone(), tile.clone(), 3, 8, 8, cut)
                .expect("the fixture loads and runs");
            let want: Vec<bool> = reference.iter().map(|&p| p >= cut).collect();
            assert_eq!(mask, want, "cut {cut}");
        }
        // A tile the model cannot accept is a Python exception, not a silently empty mask.
        assert!(cloud_mask(onnx.clone(), tile.clone(), 3, 8, 9, 0.5).is_err());
        assert!(cloud_mask(vec![0_u8; 8], tile, 3, 8, 8, 0.5).is_err());
    }

    // ── Closed-loop control ───────────────────────────────────────────────────────────────────
    // The control bindings are exercised from Python by `tests/bindings.py`, which runs against an
    // installed wheel and is therefore invisible to `cargo mutants` — the first block over this
    // section measured 2889 mutants and caught none of the ones it reached. The tests below call
    // the binding functions directly, which is the only place a mutation of their marshalling,
    // their dispatch and their refusals is observable.

    /// The moments of a body whose three principal axes are distinguishable, kg·m².
    const BODY: [f64; 9] = [4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0];
    const KP: f64 = 8.0;
    const KD: f64 = 12.0;
    const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    /// A rotation about the body x axis, row-major — deliberately not symmetric, so a transposed or
    /// zeroed `mat3_of` changes the answer rather than merely relabelling it.
    fn turn_x(angle: f64) -> [f64; 9] {
        let (s, c) = angle.sin_cos();
        [1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c]
    }

    /// The message a refused call carries, which is half of what these bindings are for: the wall a
    /// caller hits has to say which wall it was.
    fn refusal<T>(result: PyResult<T>) -> String {
        Python::attach(|py| {
            result
                .err()
                .expect("the call was expected to be refused")
                .value(py)
                .to_string()
        })
    }

    /// The two charts of one error rotation, against each other and against the angle.
    ///
    /// `σ = q_v/(1+q₀)` and `θ = 2 arccos q₀` give `‖σ‖ = tan(θ/4)` — an identity that holds for
    /// the rotation and fails for a transposed, mis-ordered or zeroed one, so it tests `mat3_of`
    /// and the two error bindings at once rather than restating what either returns.
    #[test]
    fn the_attitude_error_is_one_rotation_in_two_charts() {
        Python::attach(|_| {
            for angle in [0.05, 0.4, 1.2, 2.5] {
                let (mrp, measured) =
                    attitude_error_mrp(turn_x(angle), IDENTITY, [0.0; 3], [0.0; 3]);
                let q = attitude_error_quaternion(turn_x(angle), IDENTITY, [0.0; 3], [0.0; 3]);
                assert!(
                    (measured - angle).abs() < 1e-12,
                    "a {angle} rad turn reported {measured}"
                );
                assert!(q[0] >= 0.0, "the short way round was not taken: {q:?}");
                let norm = (mrp[0] * mrp[0] + mrp[1] * mrp[1] + mrp[2] * mrp[2]).sqrt();
                assert!(
                    (norm - (angle / 4.0).tan()).abs() < 1e-12,
                    "‖σ‖ = {norm} is not tan(θ/4) at θ = {angle}"
                );
                for i in 0..3 {
                    assert!(
                        (mrp[i] - q[i + 1] / (1.0 + q[0])).abs() < 1e-12,
                        "the two charts disagree on component {i}"
                    );
                }
            }
            // The rate arguments enter the error and nothing else: a reference rate equal to the
            // body rate is the same error as both at rest.
            let (spun, _) =
                attitude_error_mrp(turn_x(0.4), IDENTITY, [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]);
            let (still, _) = attitude_error_mrp(turn_x(0.4), IDENTITY, [0.0; 3], [0.0; 3]);
            assert_eq!(spun, still, "the error read the absolute rate");
        });
    }

    /// Each named law reaches its own arm of the dispatch, and the linearisations are the ones the
    /// crate derives — a quarter stiffness for the MRP law against a half for the quaternion one.
    ///
    /// That factor is why the two laws at equal nominal gains are different loops: the quaternion
    /// law commands the *larger* moment, and a binding that swapped the two names would still
    /// return plausible numbers.
    #[test]
    fn each_attitude_law_reaches_its_own_arm() {
        Python::attach(|_| {
            let quaternion = attitude_small_angle_gains("quaternion_pd", KP, KD)
                .expect("both gains are positive");
            let mrp = attitude_small_angle_gains("mrp_lyapunov", KP, KD)
                .expect("both gains are positive");
            assert!((quaternion.0 - 0.5 * KP).abs() < 1e-15, "{quaternion:?}");
            assert!((mrp.0 - 0.25 * KP).abs() < 1e-15, "{mrp:?}");
            assert!((quaternion.1 - KD).abs() < 1e-15 && (mrp.1 - KD).abs() < 1e-15);

            let error = turn_x(0.3);
            let hard = attitude_torque(
                "quaternion_pd",
                KP,
                KD,
                error,
                IDENTITY,
                [0.0; 3],
                [0.0; 3],
                BODY,
            )
            .expect("the law and the tensor are admissible");
            let soft = attitude_torque(
                "mrp_lyapunov",
                KP,
                KD,
                error,
                IDENTITY,
                [0.0; 3],
                [0.0; 3],
                BODY,
            )
            .expect("the law and the tensor are admissible");
            assert!(hard[0] < 0.0 && soft[0] < 0.0, "{hard:?} {soft:?}");
            assert!(
                hard[0] < soft[0],
                "the quarter-stiffness law commanded the larger moment: {hard:?} against {soft:?}"
            );

            let named = refusal(attitude_small_angle_gains("pid", KP, KD));
            assert!(
                named.contains("quaternion_pd") && named.contains("mrp_lyapunov"),
                "the refusal does not name the choices: {named}"
            );
            assert!(
                attitude_torque("pid", KP, KD, error, IDENTITY, [0.0; 3], [0.0; 3], BODY).is_err()
            );
            // A tensor that is not symmetric positive definite is not an inertia.
            assert!(
                attitude_torque(
                    "quaternion_pd",
                    KP,
                    KD,
                    error,
                    IDENTITY,
                    [0.0; 3],
                    [0.0; 3],
                    [1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                )
                .is_err()
            );
            assert!(attitude_small_angle_gains("quaternion_pd", -KP, KD).is_err());
        });
    }

    /// The MRP law's certificate: `V > 0` off the reference, `V = 0` on it, and the second-order
    /// response the quaternion law's own gains imply.
    ///
    /// `ω_n = √(K/2J)` because the linearised stiffness is `K/2`, and `P = √(2JK)` is the gain that
    /// puts `ζ = 1` — the two are the same relation read in opposite directions, so computing one
    /// from the other is a check of both.
    #[test]
    fn the_lyapunov_function_and_the_second_order_response_agree_with_their_own_gains() {
        Python::attach(|_| {
            let moment = 9.0;
            let (omega, zeta) =
                quaternion_pd_response(KP, KD, moment).expect("both gains are positive");
            assert!(
                (omega - (0.5 * KP / moment).sqrt()).abs() < 1e-15,
                "ω_n = {omega}"
            );
            assert!(
                (zeta - KD / (2.0 * moment * omega)).abs() < 1e-15,
                "ζ = {zeta}"
            );
            let critical =
                quaternion_pd_critically_damped(KP, moment).expect("the gain is positive");
            assert!(
                (critical - (2.0 * moment * KP).sqrt()).abs() < 1e-12,
                "P = {critical} is not √(2JK)"
            );
            let (_, unit) =
                quaternion_pd_response(KP, critical, moment).expect("the gain is positive");
            assert!(
                (unit - 1.0).abs() < 1e-12,
                "ζ = {unit} at the critical gain"
            );

            let rest = attitude_lyapunov(KP, KD, IDENTITY, IDENTITY, [0.0; 3], [0.0; 3], BODY)
                .expect("the gains and the tensor are admissible");
            assert!(rest.abs() < 1e-15, "V = {rest} at the reference");
            let displaced =
                attitude_lyapunov(KP, KD, turn_x(0.6), IDENTITY, [0.0; 3], [0.0; 3], BODY)
                    .expect("the gains and the tensor are admissible");
            let spinning = attitude_lyapunov(
                KP,
                KD,
                turn_x(0.6),
                IDENTITY,
                [0.3, 0.0, 0.0],
                [0.0; 3],
                BODY,
            )
            .expect("the gains and the tensor are admissible");
            assert!(displaced > 0.0, "V = {displaced} off the reference");
            assert!(
                (spinning - displaced - 0.5 * 4.0 * 0.09).abs() < 1e-12,
                "the kinetic term is not ½ δωᵀ I δω: {spinning} against {displaced}"
            );
            // The moment's domain is open at zero on both bindings: a massless axis has no
            // second-order response, and `√(K/2J)` at J = 0 is an infinity rather than a refusal.
            for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
                assert!(quaternion_pd_response(KP, KD, bad).is_err(), "moment {bad}");
                assert!(
                    quaternion_pd_critically_damped(KP, bad).is_err(),
                    "moment {bad}"
                );
            }
            assert!(
                attitude_lyapunov(0.0, KD, IDENTITY, IDENTITY, [0.0; 3], [0.0; 3], BODY).is_err()
            );
        });
    }

    /// The margin of a named law is the margin of that law's own linearisation, and the binding
    /// axis is the axis whose margin is smallest — computed, not assumed to be the smallest moment.
    #[test]
    fn the_margin_of_a_law_is_the_margin_of_its_own_linearisation() {
        Python::attach(|_| {
            let moment = 9.0;
            let by_law = delay_margin_of_law("quaternion_pd", KP, KD, moment)
                .expect("the moment is positive");
            let by_gains =
                delay_margin_of_gains(moment, 0.5 * KP, KD).expect("the gains are admissible");
            assert_eq!(
                by_law, by_gains,
                "the law's linearisation is not what the margin was taken of"
            );
            // The published form: τ_max · ω_n depends only on ζ, so the fifth entry is the product
            // of the first and the third.
            let (seconds, crossing, natural, damping, normalised) = by_law;
            assert!(
                (normalised - seconds * natural).abs() < 1e-12,
                "{normalised} is not {seconds}·{natural}"
            );
            assert!(crossing > 0.0 && damping > 0.0 && seconds > 0.0);

            let quarter = delay_margin_of_law("mrp_lyapunov", KP, KD, moment)
                .expect("the moment is positive");
            assert!(
                quarter.0 > seconds,
                "the softer law has the smaller margin: {} against {seconds}",
                quarter.0
            );

            let moments = [4.0, 9.0, 16.0];
            let (axis, worst, at) = worst_principal_axis("quaternion_pd", KP, KD, moments)
                .expect("every moment is positive");
            for (k, &j) in moments.iter().enumerate() {
                let each = delay_margin_of_law("quaternion_pd", KP, KD, j)
                    .expect("the moment is positive");
                assert!(
                    each.0 >= worst,
                    "axis {k} has margin {} below the reported worst {worst}",
                    each.0
                );
                if k == axis {
                    assert!((each.0 - worst).abs() < 1e-15 && (each.1 - at).abs() < 1e-15);
                }
            }
            assert_eq!(axis, 0, "the binding axis is not the smallest moment");
            // A zero rate gain has no margin at all, and that is the answer rather than a refusal.
            assert!(
                delay_margin_of_gains(moment, KP, 0.0)
                    .expect("admissible")
                    .0
                    == 0.0
            );
            assert!(delay_margin_of_gains(-1.0, KP, KD).is_err());
            assert!(worst_principal_axis("quaternion_pd", KP, KD, [4.0, 0.0, 16.0]).is_err());
            assert!(worst_principal_axis("pid", KP, KD, moments).is_err());
        });
    }

    /// The Clohessy–Wiltshire plant, the LQR on it, and the closed loop's own characteristic
    /// polynomial — one flat-matrix round trip checked against an analytic answer at each end.
    ///
    /// The double integrator under `Q = I`, `R = 1` has `P = [[√3, 1], [1, √3]]` and `K = [1, √3]`
    /// in closed form, so `A − BK` has characteristic polynomial `λ² + √3 λ + 1`. Three bindings,
    /// one identity: `lqr` must return that gain, `characteristic_polynomial` must return those
    /// coefficients, and `flatten`/`rows_of` must not permute anything in between.
    #[test]
    fn the_flat_matrices_survive_the_round_trip_and_the_gain_is_the_analytic_one() {
        Python::attach(|_| {
            let n = 0.001_1;
            let (a, b) = clohessy_wiltshire_plant(n).expect("the mean motion is positive");
            assert_eq!((a.len(), b.len()), (36, 18));
            // Row-major: `a[i*6 + j]`. The three couplings are what make the plant not a triple
            // integrator, and a transposed flattening moves each of them.
            assert!((a[3] - 1.0).abs() < 1e-18, "ẋ is not the fourth state");
            assert!((a[18] - 3.0 * n * n).abs() < 1e-18, "the radial spring");
            assert!((a[22] - 2.0 * n).abs() < 1e-18, "the Coriolis term");
            assert!((a[27] + 2.0 * n).abs() < 1e-18, "its antisymmetric partner");
            assert!((a[32] + n * n).abs() < 1e-18, "the cross-track spring");
            assert!((b[9] - 1.0).abs() < 1e-18 && (b[13] - 1.0).abs() < 1e-18);

            let double = vec![0.0, 1.0, 0.0, 0.0];
            let input = vec![0.0, 1.0];
            let (gain, riccati, residual) = lqr(
                double.clone(),
                input.clone(),
                vec![1.0, 0.0, 0.0, 1.0],
                vec![1.0],
            )
            .expect("the double integrator is stabilisable");
            let root3 = 3.0_f64.sqrt();
            assert!(
                (gain[0] - 1.0).abs() < 1e-9 && (gain[1] - root3).abs() < 1e-9,
                "{gain:?}"
            );
            assert!(
                (riccati[0] - root3).abs() < 1e-9
                    && (riccati[1] - 1.0).abs() < 1e-9
                    && (riccati[2] - 1.0).abs() < 1e-9
                    && (riccati[3] - root3).abs() < 1e-9,
                "{riccati:?}"
            );
            assert!(residual < 1e-12, "the solution was accepted at {residual}");

            // A − BK, with B selecting the second state: the second row loses the gain.
            let closed = vec![0.0, 1.0, -gain[0], -gain[1]];
            let poly = characteristic_polynomial(closed).expect("2×2 is inside the wall");
            assert_eq!(poly.len(), 2, "the λ² coefficient is 1 and is not returned");
            assert!(
                (poly[0] - 1.0).abs() < 1e-9 && (poly[1] - root3).abs() < 1e-9,
                "{poly:?}"
            );

            // The Clohessy–Wiltshire plant is the 6×3 arm, and its gain stabilises it.
            let mut state_cost = vec![0.0; 36];
            for i in 0..6 {
                state_cost[i * 6 + i] = 1.0;
            }
            let (cw_gain, _, cw_residual) = lqr(
                a.clone(),
                b,
                state_cost,
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            )
            .expect("the relative plant is stabilisable");
            assert_eq!(cw_gain.len(), 18, "K is 3×6");
            assert!(cw_residual < 1e-9, "{cw_residual}");
            // The closed loop is stable, and its characteristic polynomial is the 6×6 arm of the
            // same wall the graphs use: a monic degree-6 polynomial, every coefficient positive,
            // which is necessary for every root to have a negative real part.
            let mut closed_cw = vec![0.0; 36];
            for i in 0..6 {
                for j in 0..6 {
                    let mut acc = a[i * 6 + j];
                    for k in 0..3 {
                        acc -= if i == 3 + k { cw_gain[k * 6 + j] } else { 0.0 };
                    }
                    closed_cw[i * 6 + j] = acc;
                }
            }
            let cw_poly = characteristic_polynomial(closed_cw).expect("6×6 is inside the wall");
            assert_eq!(cw_poly.len(), 6);
            assert!(
                cw_poly.iter().all(|c| *c > 0.0),
                "the closed loop is not Hurwitz: {cw_poly:?}"
            );
            // A zero state cost leaves the Hamiltonian with eigenvalues on the imaginary axis, and
            // the refusal names that rather than returning an arbitrary stabilising gain.
            let undetectable = refusal(lqr(
                a,
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 1.0,
                ],
                vec![0.0; 36],
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ));
            assert!(
                undetectable.contains("ImaginaryAxisEigenvalue"),
                "{undetectable}"
            );

            // The other two monomorphisations, each with a gain the decoupling makes analytic. The
            // triple integrator is controllable, so it has one; two double integrators side by side
            // have the pair above twice, once per channel, with no cross terms.
            let triple = lqr(
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                vec![1.0],
            )
            .expect("the triple integrator is stabilisable");
            assert_eq!(triple.0.len(), 3);
            assert!(triple.2 < 1e-9, "{}", triple.2);
            #[rustfmt::skip]
            let paired = lqr(
                vec![
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0,
                ],
                vec![
                    0.0, 0.0,
                    1.0, 0.0,
                    0.0, 0.0,
                    0.0, 1.0,
                ],
                vec![
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .expect("two decoupled double integrators are stabilisable");
            for (k, w) in [1.0, root3, 0.0, 0.0, 0.0, 0.0, 1.0, root3]
                .iter()
                .enumerate()
            {
                assert!((paired.0[k] - w).abs() < 1e-8, "K = {:?}", paired.0);
            }

            assert!(
                lqr(vec![0.0; 25], input, vec![1.0; 25], vec![1.0]).is_err(),
                "(5,1) is not built"
            );
            // Both halves of the squareness check refuse on their own, and each names the pair — a
            // non-square `r` alone must not fall through to a monomorphisation and be caught later
            // by the reshape, which would answer a different question's message.
            let ragged_a = refusal(lqr(vec![0.0; 5], vec![0.0; 5], vec![1.0; 5], vec![1.0]));
            let ragged_r = refusal(lqr(
                double.clone(),
                vec![0.0, 1.0],
                vec![1.0; 4],
                vec![1.0; 2],
            ));
            for message in [&ragged_a, &ragged_r] {
                assert!(
                    message.contains("square matrix flattened row-major"),
                    "{message}"
                );
            }
            assert!(
                lqr(double, vec![0.0; 9], vec![1.0, 0.0, 0.0, 1.0], vec![1.0]).is_err(),
                "B is 2×1"
            );
        });
    }

    /// The two monomorphisation walls are the same two numbers and not the same wall, and the
    /// refusal has to say which one the caller hit.
    ///
    /// That is the whole reason `dispatch_agents!` carries a `$what` literal; without this test a
    /// mutant that swaps the two nouns, or that widens either range, is invisible.
    #[test]
    fn the_two_monomorphisation_walls_name_the_noun_the_caller_hit() {
        Python::attach(|_| {
            let big = 13;
            let mut adjacency = vec![0.0; big * big];
            for i in 0..big {
                let j = (i + 1) % big;
                adjacency[i * big + j] = 1.0;
                adjacency[j * big + i] = 1.0;
            }
            let agents = refusal(laplacian_spectrum(adjacency.clone()));
            assert!(
                agents.contains("2 to 12 agents") && agents.contains("13"),
                "the graph wall does not name agents: {agents}"
            );
            let rows = refusal(characteristic_polynomial(adjacency.clone()));
            assert!(
                rows.contains("2 to 12 rows") && rows.contains("13"),
                "the matrix wall does not name rows: {rows}"
            );
            // Twelve is inside both, so the wall is at thirteen and not at ten.
            let ring = |n: usize| {
                let mut m = vec![0.0; n * n];
                for i in 0..n {
                    let j = (i + 1) % n;
                    m[i * n + j] = 1.0;
                    m[j * n + i] = 1.0;
                }
                m
            };
            assert_eq!(
                laplacian_spectrum(ring(12))
                    .expect("twelve is inside")
                    .len(),
                12
            );
            assert_eq!(
                characteristic_polynomial(ring(12))
                    .expect("twelve is inside")
                    .len(),
                12
            );
            assert!(
                laplacian_spectrum(ring(1)).is_err(),
                "one agent cannot disagree"
            );

            // A length that is not a square is refused before the dispatch, by each function's own
            // check, and the two messages name the object the caller passed.
            let square = refusal(laplacian_spectrum(vec![0.0; 5]));
            assert!(square.contains("adjacency"), "{square}");
            let order = refusal(characteristic_polynomial(vec![0.0; 5]));
            assert!(order.contains("matrix"), "{order}");
        });
    }

    /// The spectrum, the modes, the connectivity and the delay bound are four readings of one
    /// Laplacian, and they have to agree with each other.
    ///
    /// The path graph on three agents has spectrum `{0, 1, 3}` exactly, so `λ₂ = 1`, and the
    /// Olfati-Saber bound `π/(2λ_max)` is `π/6`. The modes are checked for orthonormality and for
    /// being eigenvectors, which is a property no transcription of the eigenvalues can fake.
    #[test]
    fn the_spectrum_the_modes_and_the_bound_are_one_laplacian() {
        Python::attach(|_| {
            #[rustfmt::skip]
            let path = vec![
                0.0, 1.0, 0.0,
                1.0, 0.0, 1.0,
                0.0, 1.0, 0.0,
            ];
            let spectrum = laplacian_spectrum(path.clone()).expect("the path is connected");
            for (k, &want) in [0.0, 1.0, 3.0].iter().enumerate() {
                assert!((spectrum[k] - want).abs() < 1e-12, "{spectrum:?}");
            }
            let (values, modes) = laplacian_modes(path.clone()).expect("the path is connected");
            assert_eq!(
                values, spectrum,
                "the two bindings read different eigenvalues"
            );
            assert_eq!(modes.len(), 9);
            for k in 0..3 {
                let v = &modes[3 * k..3 * k + 3];
                let norm: f64 = v.iter().map(|x| x * x).sum();
                assert!((norm - 1.0).abs() < 1e-12, "mode {k} is not a unit vector");
                // L v = λ v, with L = D − A read from the same adjacency the binding was given.
                for i in 0..3 {
                    let degree: f64 = (0..3).map(|j| path[3 * i + j]).sum();
                    let lv: f64 =
                        degree * v[i] - (0..3).map(|j| path[3 * i + j] * v[j]).sum::<f64>();
                    assert!(
                        (lv - values[k] * v[i]).abs() < 1e-12,
                        "mode {k} is not an eigenvector at component {i}"
                    );
                }
            }
            let lambda2 = algebraic_connectivity(path.clone()).expect("the path is connected");
            assert!((lambda2 - spectrum[1]).abs() < 1e-12, "λ₂ = {lambda2}");
            assert!(graph_is_connected(path.clone()).expect("a usable graph"));

            let bound = consensus_delay_bound(path.clone(), "delayed_relative")
                .expect("the path is connected");
            assert!(
                (bound - core::f64::consts::PI / (2.0 * spectrum[2])).abs() < 1e-12,
                "τ_max = {bound}"
            );
            assert!(
                consensus_delay_bound(path, "current_self")
                    .expect("the path is connected")
                    .is_infinite(),
                "a current-self protocol has no delay bound"
            );

            // Disconnected is an answer for `graph_is_connected` and a refusal for every binding
            // whose result would be a fiction on it.
            #[rustfmt::skip]
            let split = vec![
                0.0, 0.0, 0.0,
                0.0, 0.0, 1.0,
                0.0, 1.0, 0.0,
            ];
            assert!(!graph_is_connected(split.clone()).expect("still a graph"));
            assert!(laplacian_spectrum(split.clone()).is_err());
            assert!(algebraic_connectivity(split).is_err());
            // Not symmetric is not a disconnected graph — it is not a graph, and it stays an error.
            #[rustfmt::skip]
            let skew = vec![
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
                0.0, 1.0, 0.0,
            ];
            assert!(
                graph_is_connected(skew).is_err(),
                "an asymmetric matrix was accepted"
            );
            let named = refusal(consensus_delay_bound(vec![0.0, 1.0, 1.0, 0.0], "gossip"));
            assert!(
                named.contains("delayed_relative") && named.contains("current_self"),
                "{named}"
            );
        });
    }

    /// The delayed run: it converges under the bound, its disagreement is the deviation from the
    /// mean the dynamics conserve, and its mesh is the breakpoints of its own delay.
    ///
    /// The step seed the binding hands the adaptive driver is `0.05 · τ` — proportional to the link
    /// delay, so it lands well inside the first arc of the method-of-steps mesh — and `rejected` is
    /// the only place that choice is observable. It is observable only where the driver has to
    /// reject: measured on this exact run at `τ = 0.5`, `0.05 + τ` costs 13 rejected steps and
    /// `0.05 / τ` costs 3, while all three agree on the final state to 3e-12 relative. At `τ = 0.1`
    /// all three cost none, which is why the delay here is the larger one. The seed is invisible in
    /// the answer and visible in the work, which is the only place an assertion can reach it.
    #[test]
    fn the_delayed_run_converges_under_its_own_bound() {
        Python::attach(|py| {
            let pair = vec![0.0, 1.0, 1.0, 0.0];
            let bound = consensus_delay_bound(pair.clone(), "delayed_relative")
                .expect("two agents are connected");
            assert!(
                (bound - core::f64::consts::PI / 4.0).abs() < 1e-12,
                "λ_max is 2 for a single edge, so the bound is π/4, not {bound}"
            );

            let initial = vec![1.0, -1.0];
            assert!(
                (consensus_disagreement(initial.clone()).expect("two agents")
                    - core::f64::consts::SQRT_2)
                    .abs()
                    < 1e-12
            );
            assert!(consensus_disagreement(vec![0.3, 0.3]).expect("two agents") < 1e-15);

            let delay = 0.5;
            let (final_state, disagreement, accepted, rejected) = consensus_run(
                py,
                pair.clone(),
                "delayed_relative",
                delay,
                initial,
                20.0,
                1e-9,
            )
            .expect("the delay is under the bound");
            assert!(accepted > 0, "no step was accepted");
            assert_eq!(rejected, 0, "the step seed cost {rejected} rejected steps");
            // The mean is conserved and the disagreement is driven to zero, which is consensus.
            assert!(
                (final_state[0] + final_state[1]).abs() < 1e-6,
                "the mean moved: {final_state:?}"
            );
            assert!(
                disagreement < 1e-3,
                "no consensus after 20 s: {disagreement}"
            );
            assert!(
                (disagreement - consensus_disagreement(final_state).expect("two agents")).abs()
                    < 1e-12,
                "the returned disagreement is not the state's"
            );

            assert!(
                consensus_run(
                    py,
                    pair.clone(),
                    "delayed_relative",
                    0.0,
                    vec![1.0, -1.0],
                    1.0,
                    1e-9
                )
                .is_err(),
                "a zero delay is an ordinary differential equation"
            );
            let mismatched = refusal(consensus_run(
                py,
                pair,
                "delayed_relative",
                delay,
                vec![1.0],
                1.0,
                1e-9,
            ));
            assert!(mismatched.contains("2 agents"), "{mismatched}");

            // The mesh: every sum of three or fewer delays strictly inside (0, 2).
            let mesh = delay_breakpoints(vec![0.5], 0.0, 2.0, 3).expect("the delay is positive");
            assert_eq!(mesh.len(), 3, "{mesh:?}");
            for (k, &want) in [0.5, 1.0, 1.5].iter().enumerate() {
                assert!((mesh[k] - want).abs() < 1e-12, "{mesh:?}");
            }
            assert!(delay_breakpoints(vec![-0.5], 0.0, 2.0, 3).is_err());
            assert!(delay_breakpoints(vec![0.5], 2.0, 0.0, 3).is_err());
        });
    }

    /// The station-keeping law wraps the mean-longitude error and its own Lyapunov function does
    /// not, which is the asymmetry a caller has to be able to see.
    ///
    /// A λ error of `ε + 2π` is the same orbit as `ε`, so the command and the decrease rate are
    /// bit-identical across it while `V` is not — an assertion that fails both if the wrap is
    /// dropped and if it leaks into `lyapunov`.
    #[test]
    fn the_station_keeping_command_wraps_the_longitude_and_the_certificate_does_not() {
        Python::attach(|_| {
            const MU: f64 = 3.986_004_418e14;
            let weights = [1e-6, 1.0, 1.0, 1.0, 1.0, 1.0];
            let elements = [7.0e6, 1e-3, 2e-3, 1e-4, 3e-4, 0.5];
            let error = [100.0, 1e-4, -2e-4, 0.0, 0.0, 1e-3];
            let cap = 1e-4;

            let near = element_feedback_acceleration(weights, 1.0, cap, elements, MU, 0.7, error)
                .expect("the weights and the gain are admissible");
            let mut wrapped = error;
            wrapped[5] += core::f64::consts::TAU;
            let far = element_feedback_acceleration(weights, 1.0, cap, elements, MU, 0.7, wrapped)
                .expect("the weights and the gain are admissible");
            assert_eq!(near, far, "the mean-longitude error was not wrapped");

            let v = element_feedback_lyapunov(weights, 1.0, cap, error).expect("admissible");
            let v_wrapped =
                element_feedback_lyapunov(weights, 1.0, cap, wrapped).expect("admissible");
            assert!(
                v > 0.0 && v_wrapped > v,
                "V wrapped: {v} against {v_wrapped}"
            );
            let by_hand: f64 = weights
                .iter()
                .zip(&error)
                .map(|(w, e)| 0.5 * w * e * e)
                .sum();
            assert!((v - by_hand).abs() < 1e-18, "V = {v} is not ½ ΔœᵀWΔœ");

            let rate = element_feedback_decrease_rate(weights, 1.0, cap, elements, MU, 0.7, error)
                .expect("admissible");
            assert!(rate < 0.0, "V̇ = {rate} is not a decrease");
            let magnitude = (far[0] * far[0] + far[1] * far[1] + far[2] * far[2]).sqrt();
            assert!(
                magnitude <= cap * (1.0 + 1e-12),
                "the cap was exceeded: {magnitude}"
            );
            // A zero error commands nothing and moves V nowhere, rather than dividing by a zero
            // direction.
            let idle =
                element_feedback_acceleration(weights, 1.0, cap, elements, MU, 0.7, [0.0; 6])
                    .expect("admissible");
            assert_eq!(idle, [0.0; 3]);
            assert!(
                element_feedback_decrease_rate(weights, 1.0, cap, elements, MU, 0.7, [0.0; 6])
                    .expect("admissible")
                    == 0.0
            );
            assert!(element_feedback_lyapunov([0.0; 6], 1.0, cap, error).is_err());
            assert!(element_feedback_lyapunov(weights, 1.0, 0.0, error).is_err());
        });
    }

    /// The closed loop at fixed rate under fixed latency: it holds the reference it is already at,
    /// it drives a displaced one home, and it refuses a lag its own history cannot hold.
    #[test]
    fn the_loop_holds_the_reference_and_drives_a_displaced_attitude_home() {
        Python::attach(|py| {
            let step = 0.01;
            assert!((latency_total(0.02, 0.03, 0.05).expect("all finite") - 0.1).abs() < 1e-15);
            assert_eq!(
                latency_steps(0.02, 0.03, 0.05, step).expect("a whole number"),
                10
            );
            assert!(
                latency_steps(0.02, 0.03, 0.055, step).is_err(),
                "2.5 steps is not 2 or 3"
            );
            assert!(
                latency_total(-0.01, 0.0, 0.0).is_err(),
                "a negative latency was accepted"
            );

            let held = attitude_loop_run(
                py,
                "quaternion_pd",
                KP,
                KD,
                0.01,
                0.0,
                0.0,
                step,
                IDENTITY,
                [0.0; 3],
                IDENTITY,
                [0.0; 3],
                IDENTITY,
                BODY,
                20,
            )
            .expect("one step of lag fits");
            let (times, angles, torques, rotation, rate) = held;
            assert_eq!((times.len(), angles.len(), torques.len()), (20, 20, 60));
            assert!(
                angles.iter().all(|a| a.abs() < 1e-15),
                "the loop moved off the reference"
            );
            assert!(
                torques.iter().all(|t| t.abs() < 1e-15),
                "a moment was commanded at rest"
            );
            assert!(rate.iter().all(|w| w.abs() < 1e-15));
            for (k, r) in rotation.iter().enumerate() {
                assert!((r - IDENTITY[k]).abs() < 1e-15, "{rotation:?}");
            }
            for (k, t) in times.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let want = k as f64 * step;
                assert!((t - want).abs() < 1e-12, "sample {k} is at {t}");
            }

            let start = 0.4;
            let (_, angles, torques, rotation, _) = attitude_loop_run(
                py,
                "quaternion_pd",
                KP,
                KD,
                0.01,
                0.0,
                0.0,
                step,
                turn_x(start),
                [0.0; 3],
                turn_x(start),
                [0.0; 3],
                IDENTITY,
                BODY,
                800,
            )
            .expect("one step of lag fits");
            assert!(
                (angles[0] - start).abs() < 1e-3,
                "the run did not start where it was put: {}",
                angles[0]
            );
            assert!(
                angles[799] < 0.1 * start,
                "8 s of a 1 rad/s loop left {} rad of {start}",
                angles[799]
            );
            assert!(
                torques[0] < 0.0,
                "the first moment was not restoring: {}",
                torques[0]
            );
            // The final attitude is still a rotation — the flattening did not transpose or scale it.
            for i in 0..3 {
                let norm: f64 = (0..3).map(|j| rotation[3 * i + j].powi(2)).sum();
                assert!((norm - 1.0).abs() < 1e-9, "row {i} is not a unit vector");
            }

            // A lag beyond the binding's history capacity is refused, and the message says by how
            // much rather than silently truncating the history.
            let over = refusal(attitude_loop_run(
                py,
                "quaternion_pd",
                KP,
                KD,
                6.0,
                0.0,
                0.0,
                step,
                IDENTITY,
                [0.0; 3],
                IDENTITY,
                [0.0; 3],
                IDENTITY,
                BODY,
                4,
            ));
            assert!(over.contains("600") && over.contains("512"), "{over}");
            assert!(
                attitude_loop_run(
                    py,
                    "quaternion_pd",
                    KP,
                    KD,
                    0.015,
                    0.0,
                    0.0,
                    step,
                    IDENTITY,
                    [0.0; 3],
                    IDENTITY,
                    [0.0; 3],
                    IDENTITY,
                    BODY,
                    4,
                )
                .is_err(),
                "a latency of 1.5 steps was accepted"
            );
            assert!(
                attitude_loop_run(
                    py, "pid", KP, KD, 0.01, 0.0, 0.0, step, IDENTITY, [0.0; 3], IDENTITY,
                    [0.0; 3], IDENTITY, BODY, 4,
                )
                .is_err()
            );
        });
    }
}
