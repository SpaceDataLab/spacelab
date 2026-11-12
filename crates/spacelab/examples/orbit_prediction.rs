//! The worker of `tools/orbit-pilot.py` (`plans/17` Ф30.H): fit an orbit to a day of positions under
//! one density model, freeze its drag scale, predict three days and report the error.
//!
//!     orbit_prediction --check-field FIELD FIXTURE
//!     orbit_prediction [--calibrated] SW-All.csv FIELD < jobs
//!
//! `FIELD` is a fitted SDTM-1 field as `tools/evaluate-sdtm.py export` writes it; `--check-field`
//! rebuilds it and holds it to the generator's own values at the fixture's rows, to the tolerances of
//! `tests/sdtm_reference.rs`. A job is one line, `id arc_file epoch`: the arc file holds
//! `unix latitude_deg longitude_deg altitude_m` rows, geodetic on WGS-84, from the start of the fit
//! window, and `epoch` is the Unix time the prediction starts. For each model, `sdtm` (the field) and
//! `msis` (NRLMSISE-00), and each mode, it prints one line per horizon, or `id model mode nan` when
//! the fit fails:
//!
//! ```text
//! id model mode horizon_h radial_m along_m cross_m drag_scale fit_rms_m iterations
//! ```
//!
//! **What is the same for every model.** The frame is the Earth-fixed one turned by the Earth rotation
//! angle, with UTC for UT1 and no polar motion — the frame `ScenarioAtmosphere` reads longitude in.
//! Over four days it turns against an inertial one by precession and nutation, at most about 15 m at
//! the orbit. The forces are `XGM2019e` to degree 30, the analytic Sun and Moon, a nominal cannonball
//! solar pressure and co-rotating drag with `C_d A/m` = 0.004 m²/kg times the estimated scale. The
//! Sun and Moon are read in their own J2000 axes, up to 0.3° from this frame by 2026: a few 1e-9 m/s²
//! of tide, the same under every model. The fit is seven parameters — the state at the arc's first
//! row and the drag scale — over the rows up to `epoch`, under the measured indices, first on two
//! hours, then on the whole window; the scale is then frozen.
//!
//! **The two modes.** `observed` fits and predicts under the measured indices. `forecast` reads only
//! what is known at `epoch`: the fit and the prediction use the trailing 81-day flux in place of the
//! centred one, which needs 40 days of the future, and the prediction holds the indices' state one
//! second before `epoch`. Each mode is its own fit. `--calibrated` replaces both with a third, shown
//! apart: the state and scale fitted to the whole arc under the measured indices, so the residuals
//! at the horizons are what is left once the model's density level is calibrated out.

#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::many_single_char_names
)]

use std::io::BufRead;

use spacelab_autodiff::Scalar;
use spacelab_core::{Angle, Length};
use spacelab_earth::{Geodetic, geodetic_to_itrf};
use spacelab_estimation::{GaussNewton, ResidualModel};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::ephemeris::{ASTRONOMICAL_UNIT, AnalyticMoon, AnalyticSun};
use spacelab_propagation::forces::{
    Acceleration, Atmosphere, Drag, EarthFixed, NoWind, Plain, SolarRadiationPressure, ThirdBody,
};
use spacelab_propagation::geopotential_coeffs::{MODEL_COEFFS, MODEL_GM, MODEL_RE};
use spacelab_propagation::lifetime::DecayAtmosphere;
use spacelab_propagation::lifetime_drivers::{
    DriverHistory, Scenario, ScenarioAtmosphere, earth_rotation_angle,
};
use spacelab_propagation::sdtm::coefficients::{ENVELOPE, HARMONICS, HISTORY};
use spacelab_propagation::sdtm::{
    DRIVERS, FluxExcess, NitricOxideCooling, SdtmDrivers, SdtmPoint, StormGain, StormResponse,
    Thermosphere,
};
use spacelab_propagation::space_weather::{SpaceWeatherDay, parse_space_weather};
use spacelab_propagation::variational::{ScalarSystem, Variational, augment, split};
use spacelab_propagation::{Dop853, GeoPotential, Tolerance};

const DEGREE: usize = 30;
const MU_SUN: f64 = 1.327_124_400_18e20;
const MU_MOON: f64 = 4.902_800_066e12;
const EARTH_RADIUS: f64 = 6_378_137.0;
/// A nominal cannonball: 4.56e-6 N/m² at 1 AU on `C_r A/m` = 0.005 m²/kg.
const SRP_NOMINAL: f64 = 2.28e-8;
const BALLISTIC: f64 = 0.004;
/// The Earth rotation angle's rate, rad/s: `2π·1.00273781191135448/86400`.
const ERA_RATE: f64 = 7.292_115_146_706_979e-5;
/// TT − UTC from 2017-01-01 on: 37 s of leap seconds and TT − TAI.
const TT_MINUS_UTC: f64 = 69.184;
/// 2000-01-01T12:00:00 UTC, Unix seconds.
const J2000_UNIX: f64 = 946_728_000.0;
/// Every component's weight, m. The same for every row, so it scales the covariance only, not the
/// estimate or the RMS.
const SIGMA: f64 = 50.0;
/// The fit's first stage, s; the second is the whole window.
const FIRST_STAGE: f64 = 7_200.0;
const HORIZONS_H: [f64; 5] = [6.0, 12.0, 24.0, 48.0, 72.0];
const CONTROLS: Controls = Controls::new(Tolerance::new(1e-6, 1e-12), 1e-3);
/// The radial step of the lift's scale height, m, as the NRLMSISE-00 wrapper takes it.
const LIFT_STEP: f64 = 10_000.0;

type Field = Thermosphere<HARMONICS>;

fn read_field(path: &str) -> Field {
    let text = std::fs::read_to_string(path).expect("the field file");
    let mut value = std::collections::HashMap::new();
    let mut coefficients = [[0.0; HARMONICS]; DRIVERS];
    let mut row = 0;
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let mut cells = line.split_whitespace();
        let key = cells.next().expect("a key");
        let numbers: Vec<f64> = cells.map(|c| c.parse().expect("a number")).collect();
        if key == "coefficients" {
            coefficients[row].copy_from_slice(&numbers);
            row += 1;
        } else {
            value.insert(key.to_owned(), numbers[0]);
        }
    }
    assert_eq!(row, DRIVERS, "one coefficient row per driver");
    let get = |k: &str| *value.get(k).expect(k);
    Thermosphere::from_coefficients(
        coefficients,
        get("sigma"),
        get("boundary_scale"),
        StormResponse::Cooling(NitricOxideCooling {
            heating: get("heating_s"),
            nitric_oxide: get("nitric_oxide_s"),
            saturation: get("saturation"),
        }),
    )
    .and_then(|f| {
        f.with_flux_excess(FluxExcess::Saturated {
            scale: get("excess_scale"),
        })
    })
    .and_then(|f| {
        f.with_storm_gain(StormGain::FluxPower {
            exponent: get("gain_exponent"),
        })
    })
    .expect("a usable field")
}

fn check_field(field_path: &str, fixture_path: &str) {
    let field = read_field(field_path);
    let text = std::fs::read_to_string(fixture_path).expect("the fixture");
    let (mut worst_t, mut worst_rho, mut n) = (0.0_f64, 0.0_f64, 0);
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<f64> = line
            .split(',')
            .map(|v| v.parse().expect("a number"))
            .collect();
        let point = SdtmPoint {
            altitude: f[0],
            latitude: f[1],
            local_solar_time: f[2],
            day_of_year: f[3],
        };
        let drivers = SdtmDrivers {
            f107: f[4],
            f107a: f[5],
            storm: f[6],
        };
        let t = field.exospheric_temperature(&point, &drivers);
        let rho = field.density(&point, &drivers);
        worst_t = worst_t.max((t / f[7] - 1.0).abs());
        worst_rho = worst_rho.max((rho / f[8] - 1.0).abs());
        n += 1;
    }
    println!("{n} rows: worst T_inf {worst_t:.3e}, worst rho {worst_rho:.3e} relative");
    assert!(
        n >= 32 && worst_t < 1e-12 && worst_rho < 1e-11,
        "the field does not reproduce"
    );
}

/// A [`DecayAtmosphere`] read as a drag [`Atmosphere`], with the drivers held from `segment_start`
/// and the value lifted into the STM along the radius by a scale height from 10 km higher.
struct Lifted<'a> {
    atmosphere: &'a ScenarioAtmosphere<'a, HARMONICS>,
    segment_start: f64,
}

impl Atmosphere for Lifted<'_> {
    fn density<S: Scalar>(&self, t: f64, r: &[S; 3]) -> S {
        let rf = [r[0].value(), r[1].value(), r[2].value()];
        let r0 = (rf[0] * rf[0] + rf[1] * rf[1] + rf[2] * rf[2]).sqrt();
        let up = (r0 + LIFT_STEP) / r0;
        let rho = self.atmosphere.density_held(self.segment_start, t, rf);
        let above = self.atmosphere.density_held(
            self.segment_start,
            t,
            [rf[0] * up, rf[1] * up, rf[2] * up],
        );
        let inverse_height = (rho / above).ln() / LIFT_STEP;
        let r_mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        S::from_f64(rho) * ((S::from_f64(r0) - r_mag) * S::from_f64(inverse_height)).exp()
    }
}

struct Dynamics<'a> {
    gravity: &'a EarthFixed<GeoPotential>,
    sun: ThirdBody<AnalyticSun>,
    moon: ThirdBody<AnalyticMoon>,
    srp: SolarRadiationPressure<AnalyticSun>,
    drag: Drag<Lifted<'a>, NoWind>,
}

impl ScalarSystem<7> for Dynamics<'_> {
    fn eval<S: Scalar>(&self, t: S, y: &[S; 7]) -> [S; 7] {
        let r = [y[0], y[1], y[2]];
        let v = [y[3], y[4], y[5]];
        let g = self.gravity.acceleration(t, &r, &v);
        let s = self.sun.acceleration(t, &r, &v);
        let m = self.moon.acceleration(t, &r, &v);
        let p = self.srp.acceleration(t, &r, &v);
        let d = self.drag.acceleration(t, &r, &v);
        let mut out = [S::from_f64(0.0); 7];
        out[..3].copy_from_slice(&[y[3], y[4], y[5]]);
        for i in 0..3 {
            out[3 + i] = g[i] + s[i] + m[i] + p[i] + y[6] * d[i];
        }
        out
    }
}

/// Everything about an arc that is not the orbit: the gravity field and the ephemerides, set up for
/// `t = 0` at the start of the fit window.
struct Arc {
    gravity: EarthFixed<GeoPotential>,
    sun: AnalyticSun,
    moon: AnalyticMoon,
}

impl Arc {
    fn new(start: f64, geopotential: &GeoPotential) -> Self {
        let tt = start + TT_MINUS_UTC - J2000_UNIX;
        Self {
            gravity: EarthFixed {
                inner: geopotential.clone(),
                omega: ERA_RATE,
                theta0: earth_rotation_angle(start),
            },
            sun: AnalyticSun {
                epoch_j2000_seconds: tt,
            },
            moon: AnalyticMoon {
                epoch_j2000_seconds: tt,
            },
        }
    }

    fn dynamics<'a>(
        &'a self,
        atmosphere: &'a ScenarioAtmosphere<'a, HARMONICS>,
        segment_start: f64,
    ) -> Dynamics<'a> {
        Dynamics {
            gravity: &self.gravity,
            sun: ThirdBody {
                mu: MU_SUN,
                ephemeris: self.sun,
            },
            moon: ThirdBody {
                mu: MU_MOON,
                ephemeris: self.moon,
            },
            srp: SolarRadiationPressure {
                reference_acceleration: SRP_NOMINAL,
                astronomical_unit: ASTRONOMICAL_UNIT,
                earth_radius: EARTH_RADIUS,
                sun: self.sun,
            },
            drag: Drag {
                ballistic_coefficient: BALLISTIC,
                earth_rotation_rate: ERA_RATE,
                atmosphere: Lifted {
                    atmosphere,
                    segment_start,
                },
                wind: NoWind,
            },
        }
    }

    /// `[state; STM]` at each of `times` (ascending, inside `[t0, t1]`), integrating in segments that
    /// end on the drivers' three-hourly grid so no step straddles a jump.
    fn augmented(
        &self,
        atmosphere: &ScenarioAtmosphere<'_, HARMONICS>,
        y0: &[f64; 56],
        t0: f64,
        t1: f64,
        times: &[f64],
    ) -> ([f64; 56], Vec<[f64; 56]>) {
        let (mut t, mut y, mut out, mut k) = (t0, *y0, Vec::with_capacity(times.len()), 0);
        while t < t1 {
            let end = atmosphere.next_jump(t).min(t1);
            let dynamics = self.dynamics(atmosphere, t);
            let trajectory = integrate(
                &Dop853,
                &Variational::<7, _>::new(&dynamics),
                t,
                &y,
                end,
                30.0,
                CONTROLS,
            )
            .expect("the segment integrates");
            while k < times.len() && times[k] <= end {
                out.push(trajectory.eval(times[k]));
                k += 1;
            }
            (t, y) = (end, trajectory.eval(end));
        }
        (y, out)
    }

    /// The state alone, the same way.
    fn plain(
        &self,
        atmosphere: &ScenarioAtmosphere<'_, HARMONICS>,
        y0: &[f64; 7],
        t0: f64,
        times: &[f64],
    ) -> Vec<[f64; 7]> {
        let t1 = *times.last().expect("a time");
        let (mut t, mut y, mut out, mut k) = (t0, *y0, Vec::with_capacity(times.len()), 0);
        while t < t1 {
            let end = atmosphere.next_jump(t).min(t1);
            let dynamics = self.dynamics(atmosphere, t);
            let trajectory = integrate(&Dop853, &Plain(dynamics), t, &y, end, 30.0, CONTROLS)
                .expect("the segment integrates");
            while k < times.len() && times[k] <= end {
                out.push(trajectory.eval(times[k]));
                k += 1;
            }
            (t, y) = (end, trajectory.eval(end));
        }
        out
    }
}

struct Fit<'a> {
    arc: &'a Arc,
    atmosphere: &'a ScenarioAtmosphere<'a, HARMONICS>,
    times: Vec<f64>,
    positions: Vec<[f64; 3]>,
}

impl ResidualModel<7> for Fit<'_> {
    fn residuals(&self, x: &[f64; 7], emit: &mut dyn FnMut(&[f64; 7], f64, f64)) {
        let (_, states) = self.arc.augmented(
            self.atmosphere,
            &augment::<7, 56>(x),
            0.0,
            *self.times.last().expect("a row"),
            &self.times,
        );
        let w = 1.0 / (SIGMA * SIGMA);
        for (y, p) in states.iter().zip(&self.positions) {
            let (s, phi) = split::<7, 56>(y);
            for i in 0..3 {
                emit(&phi[i], s[i] - p[i], w);
            }
        }
    }
}

/// The arc file's rows as `(Unix time, position)` in the frame described in the module docs.
fn read_arc(path: &str) -> Vec<(f64, [f64; 3])> {
    let text = std::fs::read_to_string(path).expect("the arc file");
    text.lines()
        .map(|line| {
            let f: Vec<f64> = line
                .split_whitespace()
                .map(|v| v.parse().expect("a number"))
                .collect();
            let itrf = geodetic_to_itrf(Geodetic {
                latitude: Angle::from_degrees(f[1]),
                longitude: Angle::from_degrees(f[2]),
                height: Length::from_metres(f[3]),
            })
            .vector;
            let (s, c) = earth_rotation_angle(f[0]).sin_cos();
            (
                f[0],
                [c * itrf.x - s * itrf.y, s * itrf.x + c * itrf.y, itrf.z],
            )
        })
        .collect()
}

/// A first state from the arc's first five rows: the position, and the velocity of the Lagrange
/// polynomial through them, whose error at 120 s spacing is a few dm/s.
fn first_state(rows: &[(f64, [f64; 3])]) -> [f64; 7] {
    let t: Vec<f64> = rows[..5].iter().map(|r| r.0).collect();
    let weight = |j: usize| {
        if j == 0 {
            (1..5).map(|m| 1.0 / (t[0] - t[m])).sum::<f64>()
        } else {
            (1..5)
                .filter(|&m| m != j)
                .map(|m| (t[0] - t[m]) / (t[j] - t[m]))
                .product::<f64>()
                / (t[j] - t[0])
        }
    };
    let mut x = [0.0; 7];
    x[..3].copy_from_slice(&rows[0].1);
    for j in 0..5 {
        let w = weight(j);
        for i in 0..3 {
            x[3 + i] += w * rows[j].1[i];
        }
    }
    x[6] = 1.0;
    x
}

fn ric(state: &[f64; 7], error: [f64; 3]) -> [f64; 3] {
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let unit = |a: [f64; 3]| {
        let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        [a[0] / n, a[1] / n, a[2] / n]
    };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let radial = unit(r);
    let normal = unit(cross(r, v));
    let along = cross(normal, radial);
    [dot(error, radial), dot(error, along), dot(error, normal)]
}

/// The fit over the rows up to each of `spans` in turn, so the first state's velocity error cannot
/// throw the solver off: the state at `t = 0` and the drag scale, with the per-component RMS and the
/// last stage's iterations.
fn fit(
    arc: &Arc,
    atmosphere: &ScenarioAtmosphere<'_, HARMONICS>,
    rows: &[(f64, [f64; 3])],
    spans: &[f64],
) -> Option<([f64; 7], f64, u32)> {
    // The cost carries the adaptive integrator's noise, about 1e-7 of itself, so a tighter test
    // never passes and the solver spends its iterations raising the damping on rejected steps. A
    // drop below 1e-5 of the cost leaves centimetres of fit and 1e-5 of the scale: 1.5 m at 72 h.
    let solver = GaussNewton {
        cost_tolerance: 1e-5,
        ..GaussNewton::default()
    };
    let mut x = first_state(rows);
    let mut last = None;
    for &span in spans {
        let (times, positions) = rows.iter().filter(|r| r.0 <= span).copied().unzip();
        let model = Fit {
            arc,
            atmosphere,
            times,
            positions,
        };
        let (next, _, report) = solver.solve(&model, &x).ok()?;
        x = next;
        last = Some((report, model.times.len()));
    }
    let (report, rows) = last?;
    let rms = SIGMA * (report.cost / (3 * rows) as f64).sqrt();
    (report.converged && rms.is_finite()).then_some((x, rms, report.iterations))
}

/// One arc under every model and mode: one output line per horizon, or one `nan` line where the fit
/// fails. `calibrated` fits the whole arc instead, under the measured indices, and reports the fit's
/// residuals at the horizons.
fn run_job(
    line: &str,
    geopotential: &GeoPotential,
    field: &Field,
    histories: [&DriverHistory; 2],
    calibrated: bool,
) {
    let cells: Vec<&str> = line.split_whitespace().collect();
    let (id, path, epoch) = (
        cells[0],
        cells[1],
        cells[2].parse::<f64>().expect("an epoch"),
    );
    // `t = 0` is the first row, the start of the fit window: rows need not fall on the epoch's grid.
    let rows = read_arc(path);
    let start = rows[0].0;
    let fit_end = epoch - start;
    let arc = Arc::new(start, geopotential);
    let (fit_rows, later): (Vec<_>, Vec<_>) = rows
        .into_iter()
        .map(|(t, p)| (t - start, p))
        .partition(|(t, _)| *t <= fit_end);
    let targets: Vec<(f64, f64, [f64; 3])> = HORIZONS_H
        .iter()
        .filter_map(|&h| {
            let want = fit_end + h * 3_600.0;
            later
                .iter()
                .min_by(|a, b| (a.0 - want).abs().total_cmp(&(b.0 - want).abs()))
                .filter(|(t, _)| (t - want).abs() <= 60.0)
                .map(|&(t, p)| (h, t, p))
        })
        .collect();
    let times: Vec<f64> = targets.iter().map(|t| t.1).collect();
    let [measured, trailing] = histories;
    let print = |model: &str, mode: &str, states: &[[f64; 7]], x: &[f64; 7], rms: f64, n: u32| {
        for ((hours, _, truth), p) in targets.iter().zip(states) {
            let e = ric(p, [p[0] - truth[0], p[1] - truth[1], p[2] - truth[2]]);
            println!(
                "{id}\t{model}\t{mode}\t{hours}\t{:.3}\t{:.3}\t{:.3}\t{:.6}\t{rms:.2}\t{n}",
                e[0], e[1], e[2], x[6]
            );
        }
    };
    if calibrated {
        let all: Vec<_> = fit_rows.iter().chain(&later).copied().collect();
        let end = all.last().expect("a row").0;
        for (model, field) in [("sdtm", Some((field, &ENVELOPE))), ("msis", None)] {
            let observed = ScenarioAtmosphere {
                history: measured,
                scenario: Scenario::Observed,
                epoch: start,
                field,
                tolerance: CONTROLS.tol,
            };
            let Some((x, rms, iterations)) =
                fit(&arc, &observed, &all, &[FIRST_STAGE, fit_end, end])
            else {
                println!("{id}\t{model}\tcalibrated\tnan");
                continue;
            };
            let states = arc.plain(&observed, &x, 0.0, &times);
            print(model, "calibrated", &states, &x, rms, iterations);
        }
        return;
    }
    for (model, field) in [("sdtm", Some((field, &ENVELOPE))), ("msis", None)] {
        for (mode, history) in [("observed", measured), ("forecast", trailing)] {
            let observed = ScenarioAtmosphere {
                history,
                scenario: Scenario::Observed,
                epoch: start,
                field,
                tolerance: CONTROLS.tol,
            };
            let Some((x, rms, iterations)) =
                fit(&arc, &observed, &fit_rows, &[FIRST_STAGE, fit_end])
            else {
                println!("{id}\t{model}\t{mode}\tnan");
                continue;
            };
            let prediction = ScenarioAtmosphere {
                scenario: if mode == "observed" {
                    Scenario::Observed
                } else {
                    Scenario::Persistence(history.state(epoch - 1.0))
                },
                ..observed.clone()
            };
            let at_epoch = arc.plain(&observed, &x, 0.0, &[fit_end])[0];
            let predicted = arc.plain(&prediction, &at_epoch, fit_end, &times);
            print(model, mode, &predicted, &x, rms, iterations);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--check-field") {
        check_field(&args[2], &args[3]);
        return;
    }
    let calibrated = args.get(1).map(String::as_str) == Some("--calibrated");
    let args = &args[usize::from(calibrated)..];
    let days = parse_space_weather(&std::fs::read_to_string(&args[1]).expect("the space weather"));
    let field = read_field(&args[2]);
    let measured =
        DriverHistory::from_days(&days, field.storm(), HISTORY).expect("a contiguous feed");
    // The forecast mode's feed: the centred 81-day flux needs 40 days of the future, so it is replaced
    // by the trailing one, for the fit and the prediction alike.
    let trailing_days: Vec<SpaceWeatherDay> = days
        .iter()
        .map(|d| SpaceWeatherDay {
            f107_center81: d.f107_last81,
            ..*d
        })
        .collect();
    let trailing = DriverHistory::from_days(&trailing_days, field.storm(), HISTORY)
        .expect("a contiguous feed");
    let coefficients: Vec<_> = MODEL_COEFFS
        .iter()
        .copied()
        .filter(|(n, _, _, _)| *n <= DEGREE)
        .collect();
    let geopotential = GeoPotential::from_normalized(MODEL_GM, MODEL_RE, DEGREE, &coefficients)
        .expect("a degree inside the model");
    for line in std::io::stdin().lock().lines() {
        run_job(
            &line.expect("a job"),
            &geopotential,
            &field,
            [&measured, &trailing],
            calibrated,
        );
    }
}
