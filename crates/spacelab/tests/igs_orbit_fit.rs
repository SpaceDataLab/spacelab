//! The reality check: fit an orbit to an IGS final precise ephemeris and report what is left over.
//!
//! Every other test in this workspace compares one piece against a reference for that piece. This
//! one runs the whole chain at once on a product nobody here produced — the IGS final orbit
//! combination for 2025-01-01, a weighted average of ten analysis centres in the IGS20 realization
//! of ITRF2020, quoted at the satellite centre of mass on a 15-minute grid and accurate to a few
//! centimetres. Measured Earth orientation, the CIO frame chain, the geopotential in the true
//! rotating frame, luni-solar attraction, solar radiation pressure, the Schwarzschild term, the AD
//! variational equations and the square-root batch estimator all have to be right *together* for
//! the residual to come out small, and the residual is in metres of real orbit, not in units of
//! agreement with ourselves.
//!
//! **What is estimated**: the six epoch state components and one scale on the solar-radiation
//! pressure — the classic seven-parameter GNSS arc. **What is not**: the empirical once- and
//! twice-per-revolution accelerations (ECOM) the IGS analysis centres estimate alongside, nor the
//! box-wing geometry underneath them. That is why the number here is decimetres and theirs is a few
//! centimetres, and the gap is the SRP model rather than the integrator or the frames.
//!
//! Measured on G02 over 2025-01-01, 96 observations, 3D post-fit RMS:
//!
//! | third bodies | radial | along | cross | 3D | worst |
//! |---|---|---|---|---|---|
//! | analytic Sun + Moon (the committed fixture, CI) | 0.175 | 0.221 | 0.105 | **0.300** | 0.614 |
//! | DE440 (`#[ignore]`, needs the kernel) | 0.054 | 0.063 | 0.045 | **0.094** | 0.248 |
//! | Orekit 13.1.7, its own fit of the same file | 0.053 | 0.062 | 0.065 | **0.105** | 0.252 |
//!
//! Two things are worth reading off that table. The low-precision analytic Sun and Moon cost
//! **0.21 m of the 0.30 m** — the largest single term in the residual, and not the SRP model as one
//! would guess. And with the same ephemeris quality the two independent implementations land within
//! a centimetre of each other, on solar-pressure scales that agree to 0.3% (1.0340 against 1.0310),
//! which is the actual claim this test exists to support. `tools/orekit-oracle/IgsOrbitFitOracle.java`
//! is the other half of it.
//!
//! A third thing is worth reading off the *history* of that first row, because it is the sharper
//! lesson. #306 removed the annual aberration the Almanac's series carries, cutting the analytic
//! Sun's error against ERFA `epv00` by 39 % RMS — and this fit got **worse**, 0.293 m to 0.300 m.
//! Nothing is wrong with either measurement. A more nearly geometric Sun is unambiguously the
//! correct ephemeris, but this residual is not a measurement of the solar direction: it is dominated
//! by the Moon's own `278″` and by a cannonball SRP, and seven estimated parameters had been
//! absorbing the 20.5″ bias as part of a cancellation that is now gone. A compound end-to-end number
//! moving the wrong way when a component is fixed means the component was not what it was limited
//! by — not that the fix was wrong.
//!
//! The residual is reported in the radial / along-track / cross-track frame of the fitted orbit,
//! because that is the decomposition that says *which* model is wrong: a mismodelled along-track
//! acceleration grows quadratically and dominates, radial follows the semi-major-axis error, and
//! cross-track is driven by the out-of-plane forces — luni-solar, and the Y-bias of SRP.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    // `r`, `v`, `g`, `s`, `m`, `p` are the accelerations of the force model, one letter each because
    // that is how the equations of motion are written everywhere else in this workspace.
    clippy::many_single_char_names
)]

use spacelab::{EopSource, earth_orientation_at};
use spacelab_autodiff::Scalar;
use spacelab_core::{Angle, Epoch, Ut1, UtcDateTime};
use spacelab_eop::{EopSeries, subdaily_correction};
use spacelab_estimation::{GaussNewton, ResidualModel};
use spacelab_fa::delta_at;
use spacelab_frames::{CioInterpolator, gcrf_to_itrf};
use spacelab_gnss::{Itrs, ReferenceFrame, SatId, Sp3, TimeSystem};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::ephemeris::{ASTRONOMICAL_UNIT, AnalyticMoon, AnalyticSun, Ephemeris};
use spacelab_propagation::forces::{Acceleration, BodyOriented, SolarRadiationPressure, ThirdBody};
use spacelab_propagation::geopotential_coeffs::{MODEL_COEFFS, MODEL_GM, MODEL_RE};
use spacelab_propagation::variational::{ScalarSystem, Variational, augment, split};
use spacelab_propagation::{Dop853, GeoPotential, Schwarzschild, Tolerance};

/// One GPS satellite's 24 hours out of `IGS0OPSFIN_20250010000_01D_15M_ORB.SP3`, header verbatim.
/// The header still declares the full 32-satellite constellation because it is the product's own;
/// only the `P` records were filtered. `tools/fetch-validation-data.sh` fetches the whole file.
const SP3_EXCERPT: &str = include_str!("fixtures/igs20_g02_2025001.sp3");
/// MJD 60673-60683 of the IERS `finals2000A` series — the arc plus margin, so the daily table can
/// be interpolated at both ends without touching a boundary.
const EOP_EXCERPT: &str = include_str!("fixtures/finals2000A_2025001.txt");
/// The IERS EOP 20 C04 excerpt, header verbatim: it holds the same days and names the ITRF its
/// pole is in.
const C04_EXCERPT: &str = include_str!("../../spacelab-eop/tests/fixtures/eopc04_excerpt.txt");

/// GPS time is steered to a constant 19 s behind TAI, by definition of the system.
const GPS_MINUS_TAI: f64 = -19.0;
/// Gravitational parameter of the Sun, m³/s² (IAU 2015 nominal / DE440).
const MU_SUN: f64 = 1.327_124_400_18e20;
/// Gravitational parameter of the Moon, m³/s² (DE440).
const MU_MOON: f64 = 4.902_800_066e12;
/// Earth radius for the conical shadow, m (WGS-84 / IERS mean equatorial).
const EARTH_RADIUS: f64 = 6_378_137.0;
/// Nominal cannonball solar-radiation acceleration in full sun at 1 AU, m/s². A GPS bus sits near
/// `4.56e-6 · C_r · A/m` with `A/m ≈ 0.02 m²/kg`; only the order matters, because the estimated
/// scale absorbs the rest — and a fitted scale near 1 is then a statement that it did.
const SRP_NOMINAL: f64 = 1.0e-7;
/// Geopotential truncation. The IGS analysis centres use 12×12 at GNSS altitudes; at 26 560 km the
/// degree-13 terms sit far below the SRP mismodelling that sets the floor here.
const GEOPOTENTIAL_DEGREE: usize = 12;
/// Assumed per-component uncertainty of an IGS final orbit position, m. It is a weight, not a
/// claim: with one uniform sigma the solution is plain least squares and the sigma only rescales
/// the reported covariance.
const SIGMA: f64 = 0.05;

/// The seven-parameter arc: inertial state, plus a scale on the solar-radiation pressure that is
/// constant over the arc and so has zero derivative.
///
/// The forces are summed by hand rather than through the tuple [`Acceleration`] impls because the
/// gravity term owns its frame-rotation closure and so cannot be re-composed per evaluation.
struct Dynamics<R: Fn(f64) -> [[f64; 3]; 3], Sun: Ephemeris + Copy, Moon: Ephemeris> {
    gravity: BodyOriented<GeoPotential, R>,
    sun: ThirdBody<Sun>,
    moon: ThirdBody<Moon>,
    relativity: Schwarzschild,
    srp: SolarRadiationPressure<Sun>,
}

impl<R: Fn(f64) -> [[f64; 3]; 3], Sun: Ephemeris + Copy, Moon: Ephemeris> ScalarSystem<7>
    for Dynamics<R, Sun, Moon>
{
    fn eval<S: Scalar>(&self, t: S, y: &[S; 7]) -> [S; 7] {
        let r = [y[0], y[1], y[2]];
        let v = [y[3], y[4], y[5]];
        let g = self.gravity.acceleration(t, &r, &v);
        let s = self.sun.acceleration(t, &r, &v);
        let m = self.moon.acceleration(t, &r, &v);
        let rel = self.relativity.acceleration(t, &r, &v);
        let p = self.srp.acceleration(t, &r, &v);
        [
            y[3],
            y[4],
            y[5],
            g[0] + s[0] + m[0] + rel[0] + y[6] * p[0],
            g[1] + s[1] + m[1] + rel[1] + y[6] * p[1],
            g[2] + s[2] + m[2] + rel[2] + y[6] * p[2],
            S::from_f64(0.0),
        ]
    }
}

/// One SP3 sample, already rotated into GCRF.
struct Observation {
    /// Seconds from the arc epoch.
    t: f64,
    /// Inertial position, m.
    position: [f64; 3],
}

/// Propagate the current epoch estimate with its state-transition matrix and emit, per observation
/// component, the row `∂rᵢ(t)/∂x₀` straight out of the STM — a position observation *is* the state,
/// so no measurement partial stands between the two.
struct ArcFit<'a, R: Fn(f64) -> [[f64; 3]; 3], Sun: Ephemeris + Copy, Moon: Ephemeris> {
    dynamics: &'a Dynamics<R, Sun, Moon>,
    observations: &'a [Observation],
    span: f64,
}

impl<R: Fn(f64) -> [[f64; 3]; 3], Sun: Ephemeris + Copy, Moon: Ephemeris> ArcFit<'_, R, Sun, Moon> {
    /// The augmented `[state; STM]` solution over the arc, as a function of time.
    fn propagate(&self, x: &[f64; 7]) -> impl Fn(f64) -> [f64; 56] + use<'_, R, Sun, Moon> {
        let var = Variational::<7, _>::new(self.dynamics);
        let aug0: [f64; 56] = augment::<7, 56>(x);
        let traj = integrate(
            &Dop853,
            &var,
            0.0,
            &aug0,
            self.span,
            60.0,
            Controls::new(Tolerance::new(1e-6, 1e-12), 1e-3),
        )
        .expect("the arc integrates");
        move |t| traj.eval(t)
    }
}

impl<R: Fn(f64) -> [[f64; 3]; 3], Sun: Ephemeris + Copy, Moon: Ephemeris> ResidualModel<7>
    for ArcFit<'_, R, Sun, Moon>
{
    fn residuals(&self, x: &[f64; 7], emit: &mut dyn FnMut(&[f64; 7], f64, f64)) {
        let at = self.propagate(x);
        let weight = 1.0 / (SIGMA * SIGMA);
        for o in self.observations {
            let (state, phi) = split::<7, 56>(&at(o.t));
            for i in 0..3 {
                emit(&phi[i], state[i] - o.position[i], weight);
            }
        }
    }
}

/// Radial / along-track / cross-track unit vectors of a state, the frame the residual is read in.
fn rtn_axes(state: &[f64; 7]) -> [[f64; 3]; 3] {
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let radial = unit(r);
    let cross = unit(cross_product(r, v));
    [radial, cross_product(cross, radial), cross]
}

fn cross_product(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Everything the arc needs that is a property of the epoch rather than of the orbit.
struct Arc {
    observations: Vec<Observation>,
    cio: CioInterpolator,
    /// Seconds past J2000 TT of the arc epoch, for the analytic ephemerides.
    epoch_tt_seconds: f64,
    span: f64,
}

/// The `finals2000A` excerpt, the series every fit in this file is measured on.
fn finals() -> EopSeries {
    EopSeries::parse_finals2000a(EOP_EXCERPT).expect("the EOP excerpt parses")
}

/// Read an SP3, rotate one satellite's ITRF samples into GCRF with measured Earth orientation, and
/// build the epoch-dependent context the force model needs.
///
/// The observations are rotated with the **exact** per-epoch transform and per-epoch EOP, including
/// the IERS sub-daily ocean-tide and libration terms: over a day the polar motion moves ~1 mas and
/// `ΔUT1` a few tenths of a millisecond, which at 26 560 km is decimetres to metres — the size of
/// the answer. The force model's Earth-fixed rotation goes through [`CioInterpolator`], which holds
/// the EOP fixed over the span, because there it multiplies only the *tesseral* field: a metre of
/// frame error moves a 10⁻¹² m/s² term, not the 0.57 m/s² central one.
fn build_arc(sp3_text: &str, series: &EopSeries, sat: SatId) -> Arc {
    let sp3 = Sp3::parse(sp3_text);
    assert_eq!(
        sp3.time_system(),
        TimeSystem::Gps,
        "an IGS product is stamped in GPS time"
    );
    // The frame the products claim, checked rather than asserted in prose. `spacelab_gnss` now
    // surfaces line 1's coordinate-system field, and an SP3 in another datum -- an IAC GLONASS
    // product is PZ-90, decimetres from ITRF -- would otherwise be differenced against ITRF
    // coordinates with nothing to show for it.
    assert!(
        sp3.reference_frame()
            .agrees_with(ReferenceFrame::Itrs(Itrs::Itrf2020)),
        "the IGS combination should realize ITRF2020; the file says {:?}",
        sp3.reference_frame()
    );
    let start_gps = sp3.start_epoch().expect("the file carries epochs");
    let epochs = sp3.epochs().to_vec();
    let samples = sp3
        .positions(sat)
        .expect("the file carries the satellite")
        .to_vec();
    let span = *epochs.last().expect("more than one epoch");

    // GPS labels and UTC labels are the same calendar arithmetic offset by the leap-second count,
    // so the UTC instant is the GPS one shifted by (TAI−UTC) − 19 s. Resolve that count from an
    // approximate UTC first: it can only be wrong within 18 s of midnight on a leap-second day, and
    // the table is stepwise constant over whole months either side.
    let approx = UtcDateTime::from_naive_ns_since_j2000((start_gps * 1e9).round() as i128, false);
    let gps_minus_utc =
        f64::from(delta_at(approx).expect("a tabulated leap-second count")) + GPS_MINUS_TAI;

    let utc_at = |t: f64| {
        UtcDateTime::from_naive_ns_since_j2000(
            ((start_gps + t - gps_minus_utc) * 1e9).round() as i128,
            false,
        )
    };

    let mut observations = Vec::with_capacity(samples.len());
    for (&t, sample) in epochs.iter().zip(&samples) {
        let Some(itrf) = *sample else { continue };
        let utc = utc_at(t);
        let e = earth_orientation_at(utc, EopSource::Series(series))
            .expect("the arc is inside the EOP excerpt");
        let sd = subdaily_correction(utc.modified_julian_date());
        let ut1 = Epoch::<Ut1>::from_ns_since_j2000(
            e.orientation.ut1.ns_since_j2000() + (sd.ddut1_seconds * 1e9).round() as i128,
        );
        let m = gcrf_to_itrf(
            ut1,
            e.orientation.tt,
            Angle::from_radians(e.orientation.xp.radians() + sd.dxp.radians()),
            Angle::from_radians(e.orientation.yp.radians() + sd.dyp.radians()),
            // The finals2000A celestial-pole offsets are under a milliarcsecond, which is 13 cm at
            // this radius — below the SRP-driven floor, so the columns are left unread rather than
            // half-read.
            Angle::from_radians(0.0),
            Angle::from_radians(0.0),
            e.orientation.lod_seconds + sd.dlod_seconds,
        )
        .rotation
        .matrix()
        .rows;
        // GCRF = Mᵀ · ITRF.
        let position = [
            m[0][0] * itrf[0] + m[1][0] * itrf[1] + m[2][0] * itrf[2],
            m[0][1] * itrf[0] + m[1][1] * itrf[1] + m[2][1] * itrf[2],
            m[0][2] * itrf[0] + m[1][2] * itrf[1] + m[2][2] * itrf[2],
        ];
        observations.push(Observation { t, position });
    }

    let epoch = earth_orientation_at(utc_at(0.0), EopSource::Series(series))
        .expect("the epoch is in range");
    let cio = CioInterpolator::new(
        epoch.orientation.ut1,
        epoch.orientation.tt,
        span,
        epoch.orientation.xp,
        epoch.orientation.yp,
        Angle::from_radians(0.0),
        Angle::from_radians(0.0),
        epoch.orientation.lod_seconds,
    );
    let epoch_tt_seconds = epoch.orientation.tt.ns_since_j2000() as f64 / 1e9;

    Arc {
        observations,
        cio,
        epoch_tt_seconds,
        span,
    }
}

/// Post-fit residual statistics, metres.
struct Residuals {
    radial: f64,
    along: f64,
    cross: f64,
    total: f64,
    worst: f64,
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

fn residual_statistics<R: Fn(f64) -> [[f64; 3]; 3], Sun: Ephemeris + Copy, Moon: Ephemeris>(
    fit: &ArcFit<'_, R, Sun, Moon>,
    x: &[f64; 7],
) -> Residuals {
    let at = fit.propagate(x);
    let (mut radial, mut along, mut cross, mut total) = (vec![], vec![], vec![], vec![]);
    let mut worst = 0.0_f64;
    for o in fit.observations {
        let (state, _) = split::<7, 56>(&at(o.t));
        let d = [
            state[0] - o.position[0],
            state[1] - o.position[1],
            state[2] - o.position[2],
        ];
        let axes = rtn_axes(&state);
        radial.push(dot(d, axes[0]));
        along.push(dot(d, axes[1]));
        cross.push(dot(d, axes[2]));
        let norm = dot(d, d).sqrt();
        total.push(norm);
        worst = worst.max(norm);
    }
    Residuals {
        radial: rms(&radial),
        along: rms(&along),
        cross: rms(&cross),
        total: rms(&total),
        worst,
    }
}

/// The epoch velocity from the first five samples by the fourth-order one-sided difference
/// `(−25p₀ + 48p₁ − 36p₂ + 16p₃ − 3p₄)/12h`.
///
/// Its truncation error at a 900 s step on a GNSS orbit is ~0.2 m/s, tens of kilometres by the end
/// of the arc. That is an initial guess, not an answer: it is well inside the linear regime the
/// state-transition matrix describes, which is all Gauss-Newton needs to start.
fn initial_velocity(observations: &[Observation]) -> [f64; 3] {
    let h = observations[1].t - observations[0].t;
    let p = |k: usize, i: usize| observations[k].position[i];
    let mut v = [0.0; 3];
    for i in 0..3 {
        v[i] = (-25.0 * p(0, i) + 48.0 * p(1, i) - 36.0 * p(2, i) + 16.0 * p(3, i) - 3.0 * p(4, i))
            / (12.0 * h);
    }
    v
}

/// Solve the arc with the analytic Sun and Moon — the ephemerides that need no data file, and so
/// the ones the committed fixture can be measured with in CI.
fn fit_arc(arc: &Arc) -> ([f64; 7], [f64; 7], Residuals, u32) {
    fit_arc_with(
        arc,
        AnalyticSun {
            epoch_j2000_seconds: arc.epoch_tt_seconds,
        },
        AnalyticMoon {
            epoch_j2000_seconds: arc.epoch_tt_seconds,
        },
    )
}

/// Solve the arc: the estimate, its formal sigmas, the post-fit residuals and the iteration count.
fn fit_arc_with<Sun: Ephemeris + Copy, Moon: Ephemeris>(
    arc: &Arc,
    sun: Sun,
    moon: Moon,
) -> ([f64; 7], [f64; 7], Residuals, u32) {
    let coefficients: Vec<_> = MODEL_COEFFS
        .iter()
        .copied()
        .filter(|(n, _, _, _)| *n <= GEOPOTENTIAL_DEGREE)
        .collect();
    let geopotential =
        GeoPotential::from_normalized(MODEL_GM, MODEL_RE, GEOPOTENTIAL_DEGREE, &coefficients)
            .expect("a degree well inside the model");
    let dynamics = Dynamics {
        gravity: BodyOriented {
            inner: geopotential,
            rotate: |t: f64| arc.cio.rotation_rows(t),
        },
        sun: ThirdBody {
            mu: MU_SUN,
            ephemeris: sun,
        },
        moon: ThirdBody {
            mu: MU_MOON,
            ephemeris: moon,
        },
        relativity: Schwarzschild { mu: MODEL_GM },
        srp: SolarRadiationPressure {
            reference_acceleration: SRP_NOMINAL,
            astronomical_unit: ASTRONOMICAL_UNIT,
            earth_radius: EARTH_RADIUS,
            sun,
        },
    };
    let fit = ArcFit {
        dynamics: &dynamics,
        observations: &arc.observations,
        span: arc.span,
    };

    let v0 = initial_velocity(&arc.observations);
    let mut x0 = [0.0; 7];
    x0[..3].copy_from_slice(&arc.observations[0].position);
    x0[3..6].copy_from_slice(&v0);
    x0[6] = 1.0;

    let solver = GaussNewton {
        max_iterations: 30,
        ..GaussNewton::default()
    };
    let (x, covariance, report) = solver.solve(&fit, &x0).expect("the arc is observable");
    assert!(
        report.converged,
        "the batch solve stopped at the iteration cap: {report:?}"
    );
    let mut sigmas = [0.0; 7];
    for i in 0..7 {
        sigmas[i] = covariance[i][i].sqrt();
    }
    (x, sigmas, residual_statistics(&fit, &x), report.iterations)
}

/// The measurement. The numbers in the assertions are the ones observed, rounded outward.
#[test]
fn the_igs_final_orbit_is_fitted_to_decimetres_by_a_seven_parameter_arc() {
    let arc = build_arc(
        SP3_EXCERPT,
        &finals(),
        SatId {
            system: 'G',
            prn: 2,
        },
    );
    assert_eq!(arc.observations.len(), 96, "24 h on a 15-minute grid");
    assert!((arc.span - 85_500.0).abs() < 1e-9);

    let (x, sigmas, r, iterations) = fit_arc(&arc);

    println!(
        "G02 2025-01-01, {} observations, {iterations} iterations\n\
         post-fit RMS  radial {:.3} m  along-track {:.3} m  cross-track {:.3} m\n\
         3D RMS {:.3} m, worst {:.3} m\n\
         SRP scale {:.4} ± {:.4} (nominal {SRP_NOMINAL:e} m/s²)\n\
         epoch position sigma {:.4} m, velocity sigma {:.6} m/s",
        arc.observations.len(),
        r.radial,
        r.along,
        r.cross,
        r.total,
        r.worst,
        x[6],
        sigmas[6],
        sigmas[0],
        sigmas[3],
    );

    // The scale is estimated, not assumed, so it is free to come back absurd if the SRP geometry or
    // the shadow model is wrong. A GPS bus is within a factor of a few of the nominal.
    assert!(
        (0.2..5.0).contains(&x[6]),
        "the fitted SRP scale left the physical range: {}",
        x[6]
    );
    // Measured: radial 0.175, along-track 0.221, cross-track 0.105, 3D RMS 0.300, worst 0.614 m.
    // Decimetres, not centimetres — a cannonball with one scale factor is not ECOM — and the bound
    // is set just above what was seen, because the failure this guards against is not a slow drift
    // but a frame, time-scale or force-sign error, and those are kilometres.
    assert!(r.total < 0.4, "3D post-fit RMS {:.3} m", r.total);
    assert!(r.radial < 0.25, "radial RMS {:.3} m", r.radial);
    assert!(
        r.worst < 1.0,
        "worst single-epoch residual {:.3} m",
        r.worst
    );
    // Along-track is the largest component whenever the along-track acceleration is mismodelled,
    // and with a cannonball SRP it is.
    assert!(
        r.along >= r.cross,
        "along {:.3} cross {:.3}",
        r.along,
        r.cross
    );
}

/// The same arc on the IERS EOP 20 C04 series, which names its frame where `finals2000A` cannot.
///
/// The SP3 says IGS20 and the C04 header says ITRF2020, so the orbit and the Earth-orientation
/// series that rotates it are now checked to be in one frame rather than said to be. Over these
/// eleven days the two series differ by at most 0.05 mas in the pole and not at all in `ΔUT1` or
/// `dX, dY` -- 6 mm at this radius -- so the fit must not move.
#[test]
fn the_c04_series_names_the_frame_the_orbit_is_in() {
    let c04 = EopSeries::parse_c04(C04_EXCERPT).expect("the C04 excerpt parses");
    let sp3 = Sp3::parse(SP3_EXCERPT);
    assert!(
        sp3.reference_frame().agrees_with(c04.frame()),
        "orbit {:?}, Earth orientation {:?}",
        sp3.reference_frame(),
        c04.frame()
    );
    assert!(
        !sp3.reference_frame().agrees_with(finals().frame()),
        "finals2000A states no frame, so it agrees with none"
    );

    let sat = SatId {
        system: 'G',
        prn: 2,
    };
    let (_, _, on_c04, _) = fit_arc(&build_arc(SP3_EXCERPT, &c04, sat));
    let (_, _, on_finals, _) = fit_arc(&build_arc(SP3_EXCERPT, &finals(), sat));
    let moved = (on_c04.total - on_finals.total).abs();
    println!(
        "3D RMS on C04 {:.4} m, on finals2000A {:.4} m, moved {moved:.2e} m",
        on_c04.total, on_finals.total
    );
    assert!(
        moved < 1e-3,
        "the fit moved {moved:.2e} m between the two series"
    );
}

/// The whole 32-satellite product, when it has been fetched, on DE440 third bodies.
///
/// One line per satellite, so the spread across the constellation — and any satellite in an eclipse
/// season, where a cannonball SRP is at its worst — is visible rather than averaged away. The DE440
/// ephemeris is what makes this table directly comparable to
/// `tools/orekit-oracle/IgsOrbitFitOracle.java`, which reads the same kernel.
#[test]
#[ignore = "needs snapshots from tools/fetch-validation-data.sh and tools/fetch-ephemeris.sh"]
fn every_gps_satellite_in_the_daily_product() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("../../data/validation/IGS0OPSFIN_20250010000_01D_15M_ORB.SP3");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "run tools/fetch-validation-data.sh first ({}): {e}",
            path.display()
        )
    });
    let kernel_path = root.join("../../data/ephemeris/de440s.bsp");
    let kernel = spacelab_ephemeris::Ephemeris::load(kernel_path.to_str().expect("a UTF-8 path"))
        .unwrap_or_else(|e| panic!("run tools/fetch-ephemeris.sh first: {e}"));
    let sp3 = Sp3::parse(&text);

    println!("sat    radial    along    cross       3D     worst   srp_scale");
    let mut worst_total = 0.0_f64;
    let mut total = 0.0_f64;
    let mut fitted = 0_u32;
    for sat in sp3.satellites().collect::<Vec<_>>() {
        let arc = build_arc(&text, &finals(), sat);
        if arc.observations.len() < 90 {
            println!(
                "{}{:02}  skipped, {} samples",
                sat.system,
                sat.prn,
                arc.observations.len()
            );
            continue;
        }
        let (x, _, r, _) = fit_arc_with(
            &arc,
            kernel.perturbing_body(
                spacelab_ephemeris::frames::SUN_J2000,
                spacelab_ephemeris::frames::EARTH_J2000,
                arc.epoch_tt_seconds,
            ),
            kernel.perturbing_body(
                spacelab_ephemeris::frames::MOON_J2000,
                spacelab_ephemeris::frames::EARTH_J2000,
                arc.epoch_tt_seconds,
            ),
        );
        println!(
            "{}{:02}  {:8.3} {:8.3} {:8.3} {:8.3} {:9.3}   {:7.4}",
            sat.system, sat.prn, r.radial, r.along, r.cross, r.total, r.worst, x[6]
        );
        worst_total = worst_total.max(r.total);
        total += r.total;
        fitted += 1;
    }
    println!(
        "mean 3D RMS {:.3} m over {fitted} satellites",
        total / f64::from(fitted)
    );
    // Measured across all 32 GPS satellites on 2025-01-01: mean 3D RMS 0.249 m, best G17 at 0.093,
    // worst G31 at 0.612. The spread is the spread of bus types and beta angles against one
    // cannonball — the same G31 is worst in Orekit's independent fit, so it is the satellite and not
    // the implementation. Per satellite the two agree within 11 mm on 31 of the 32; the exception is
    // G19, where Orekit's solve lands 104 mm worse than ours.
    assert!(
        worst_total < 0.8,
        "worst 3D RMS across the constellation {worst_total:.3} m"
    );
}

/// Where the rest of the residual goes: the analytic Sun and Moon, replaced by DE440.
///
/// The analytic series are documented at `≤ 0.0153°` for the Sun and `≤ 278″` and `≤ 458 km` for
/// the Moon. On a 26 560 km orbit the Moon's error alone moves its `4.6e-6 m/s²` pull by a few
/// `1e-9 m/s²`, and a constant-ish acceleration error over a day is metres — most of which the
/// seven estimated parameters absorb, and the part that does not is what this measures. Same file,
/// same frames, same estimator; only the third bodies change.
#[test]
#[ignore = "needs data/ephemeris/de440s.bsp from tools/fetch-ephemeris.sh"]
fn de440_third_bodies_are_worth_the_larger_half_of_the_residual() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/ephemeris/de440s.bsp");
    let kernel = spacelab_ephemeris::Ephemeris::load(path.to_str().expect("a UTF-8 path"))
        .unwrap_or_else(|e| panic!("run tools/fetch-ephemeris.sh first: {e}"));

    let arc = build_arc(
        SP3_EXCERPT,
        &finals(),
        SatId {
            system: 'G',
            prn: 2,
        },
    );
    let (_, _, analytic, _) = fit_arc(&arc);
    let (x, _, de440, _) = fit_arc_with(
        &arc,
        kernel.perturbing_body(
            spacelab_ephemeris::frames::SUN_J2000,
            spacelab_ephemeris::frames::EARTH_J2000,
            arc.epoch_tt_seconds,
        ),
        kernel.perturbing_body(
            spacelab_ephemeris::frames::MOON_J2000,
            spacelab_ephemeris::frames::EARTH_J2000,
            arc.epoch_tt_seconds,
        ),
    );

    println!(
        "G02 2025-01-01, third bodies analytic vs DE440\n\
         analytic  radial {:.3}  along {:.3}  cross {:.3}  3D {:.3}  worst {:.3}\n\
         DE440     radial {:.3}  along {:.3}  cross {:.3}  3D {:.3}  worst {:.3}\n\
         SRP scale {:.4}",
        analytic.radial,
        analytic.along,
        analytic.cross,
        analytic.total,
        analytic.worst,
        de440.radial,
        de440.along,
        de440.cross,
        de440.total,
        de440.worst,
        x[6],
    );

    // Measured: 0.094 m against the analytic series' 0.300 m — a factor of 3.2, and enough to put
    // the fit a centimetre *ahead* of Orekit's own 0.105 m on the same file.
    assert!(
        de440.total < 0.15,
        "DE440 3D post-fit RMS {:.3} m",
        de440.total
    );
    assert!(
        de440.total < 0.5 * analytic.total,
        "DE440 {:.3} m did not halve the analytic series' {:.3} m",
        de440.total,
        analytic.total
    );
}
