//! Does the horizontal density gradient that SDTM-1's drag wrapper drops move an orbit fit?
//! (`plans/17` Ф23.R.)
//!
//!     cargo run -p spacelab --release --example sdtm_horizontal_gradient
//!
//! `SdtmAtmosphere` lifts the density into the state-transition matrix as `ρ₀·exp((|r₀| − |r|)/H)`,
//! so the STM sees `∂ρ/∂z·r̂` and nothing of the gradient across the ground. This example fits the
//! same observations twice, with that lift and with the full gradient, and asks whether the two
//! estimates differ by more than the fit's own interval. It touches no file the D-30 freeze pins.
//!
//! **The experiment.** Truth is a 24 h arc integrated under point mass, `J₂` and co-rotating drag
//! with NRLMSISE-00 density, so the fitted model is wrong in the way an operational one is. The
//! orbit is circular and near-polar (87.4°) with its node at local noon at an equinox: it crosses
//! the day–night contrast twice a revolution, the largest horizontal gradient the model has, so a
//! NO-GO here bounds the gentler geometries. Observations are positions every 30 s with 5 cm
//! Gaussian noise. The model is the same forces with SDTM-1's shipped field and an estimated drag
//! scale — seven parameters, as `tests/igs_orbit_fit.rs` estimates seven. Both fits start from the
//! true epoch state and evaluate the identical density value; only the Jacobian differs:
//!
//! - **radial:** the shipped `SdtmAtmosphere`;
//! - **full:** `FullGradient` around it, whose lift is `ρ₀·exp(∇ρ·(r − r₀)/ρ₀)`, with `∇ρ` the central
//!   difference (10 m) of the f64 field through the same geodetic closure.
//!
//! The instrument is checked first: `∇ρ·n̂`, with `n̂` the ellipsoid normal, must equal the field's
//! analytic `∂ρ/∂z`, because geodetic latitude and longitude are constant along the normal.
//!
//! **Verdict rule, fixed before the first run.** The interval of an estimate is its formal sigma
//! from the full fit, scaled by the post-fit variance factor `cost/(m − 7)`. GO — derive the
//! analytic full gradient in Maxima, ship it, and re-baseline the freeze — only if, in some scenario,
//! some estimated parameter differs between the two fits by more than its interval. Otherwise NO-GO:
//! the frozen wrapper stays, and the measured gradient ratio is recorded.
//!
//! The result, NO-GO with the largest move 3.0e-3 of its interval, is `docs/verification.md` §10.54.

#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::many_single_char_names
)]

use spacelab_autodiff::Scalar;
use spacelab_core::Vec3;
use spacelab_earth::itrf_to_geodetic;
use spacelab_estimation::{GaussNewton, ResidualModel};
use spacelab_frames::{EARTH_ROTATION_RATE, Itrf, Position};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::forces::{Acceleration, Atmosphere, Drag, J2, NoWind, PointMass};
use spacelab_propagation::sdtm::{SdtmAtmosphere, SdtmDrivers, SdtmPoint, coefficients};
use spacelab_propagation::variational::{ScalarSystem, Variational, augment, split};
use spacelab_propagation::{Dop853, Nrlmsise00Atmosphere, Tolerance};

/// EGM2008, as `cpf_mean_state` and the lifetime campaign take it.
const MU: f64 = 3.986_004_415e14;
const RE: f64 = 6_378_136.3;
/// `C̄₂₀` of EGM2008, tide-free; `J₂ = −√5·C̄₂₀`.
const C20_BAR: f64 = -0.484_165_143_790_815e-3;
/// `C_d A/m`, m²/kg: 2.2 on 0.5 m² over 100 kg.
const BALLISTIC: f64 = 0.011;
const INCLINATION_DEG: f64 = 87.4;
/// Day of year at the start of the epoch's day (21 March), and the epoch's UT: noon.
const DOY0: i32 = 80;
const SEC0: f64 = 43_200.0;
const SPAN: f64 = 86_400.0;
const CADENCE: f64 = 30.0;
/// `SPAN / CADENCE + 1`: both ends of the arc.
const OBSERVATIONS: usize = 2_881;
const SIGMA: f64 = 0.05;
/// Central-difference step of the full gradient, m.
const STEP: f64 = 10.0;

/// Inertial to Earth-fixed by the Earth rotation angle `ω⊕ t` from zero at the epoch; the drag
/// force's co-rotation is about the same axis at the same rate.
fn earth_fixed(t: f64, r: [f64; 3]) -> [f64; 3] {
    let (s, c) = (EARTH_ROTATION_RATE * t).sin_cos();
    [c * r[0] + s * r[1], -s * r[0] + c * r[1], r[2]]
}

/// The closure both atmospheres take: `(altitude_km, geodetic latitude°, longitude°)` on WGS-84.
fn geodetic(t: f64, r: [f64; 3]) -> (f64, f64, f64) {
    let e = earth_fixed(t, r);
    let g = itrf_to_geodetic(Position::<Itrf>::new(Vec3::new(e[0], e[1], e[2])))
        .expect("a LEO position is outside the singular lens");
    (
        g.height.metres() / 1_000.0,
        g.latitude.degrees(),
        g.longitude.degrees(),
    )
}

/// The ellipsoid normal at `r`, inertial.
fn normal(t: f64, r: [f64; 3]) -> [f64; 3] {
    let (_, lat, lon) = geodetic(t, r);
    let (sl, cl) = lat.to_radians().sin_cos();
    let (so, co) = lon.to_radians().sin_cos();
    let n = [cl * co, cl * so, sl];
    let (s, c) = (EARTH_ROTATION_RATE * t).sin_cos();
    [c * n[0] - s * n[1], s * n[0] + c * n[1], n[2]]
}

/// The gradient of the f64 density through the geodetic closure, by central differences.
fn gradient<A: Atmosphere>(a: &A, t: f64, r: [f64; 3]) -> [f64; 3] {
    let mut g = [0.0; 3];
    for i in 0..3 {
        let (mut up, mut dn) = (r, r);
        up[i] += STEP;
        dn[i] -= STEP;
        g[i] = (a.density(t, &up) - a.density(t, &dn)) / (2.0 * STEP);
    }
    g
}

/// The same density value as the wrapped model, lifted along the full gradient instead of the
/// radial one.
struct FullGradient<A>(A);

impl<A: Atmosphere> Atmosphere for FullGradient<A> {
    fn density<S: Scalar>(&self, t: f64, r: &[S; 3]) -> S {
        let r0 = [r[0].value(), r[1].value(), r[2].value()];
        let rho0 = self.0.density(t, &r0);
        let g = gradient(&self.0, t, r0);
        let mut e = S::from_f64(0.0);
        for i in 0..3 {
            e = e + S::from_f64(g[i] / rho0) * (r[i] - S::from_f64(r0[i]));
        }
        S::from_f64(rho0) * e.exp()
    }
}

/// Position, velocity and a drag scale, constant over the arc.
struct Dynamics<A: Atmosphere> {
    gravity: (PointMass, J2),
    drag: Drag<A, NoWind>,
}

impl<A: Atmosphere> ScalarSystem<7> for Dynamics<A> {
    fn eval<S: Scalar>(&self, t: S, y: &[S; 7]) -> [S; 7] {
        let r = [y[0], y[1], y[2]];
        let v = [y[3], y[4], y[5]];
        let g = self.gravity.acceleration(t, &r, &v);
        let d = self.drag.acceleration(t, &r, &v);
        [
            y[3],
            y[4],
            y[5],
            g[0] + y[6] * d[0],
            g[1] + y[6] * d[1],
            g[2] + y[6] * d[2],
            S::from_f64(0.0),
        ]
    }
}

fn dynamics<A: Atmosphere>(atmosphere: A) -> Dynamics<A> {
    Dynamics {
        gravity: (
            PointMass { mu: MU },
            J2 {
                mu: MU,
                re: RE,
                j2: -C20_BAR * 5.0_f64.sqrt(),
            },
        ),
        drag: Drag {
            ballistic_coefficient: BALLISTIC,
            earth_rotation_rate: EARTH_ROTATION_RATE,
            atmosphere,
            wind: NoWind,
        },
    }
}

/// `[state; STM]` at each requested time.
fn propagate<A: Atmosphere>(
    d: &Dynamics<A>,
    x: &[f64; 7],
    tol: Tolerance,
) -> impl Fn(f64) -> [f64; 56] {
    let traj = integrate(
        &Dop853,
        &Variational::<7, _>::new(d),
        0.0,
        &augment::<7, 56>(x),
        SPAN,
        30.0,
        Controls::new(tol, 1e-3),
    )
    .expect("the arc integrates");
    move |t| traj.eval(t)
}

const FIT_TOLERANCE: Tolerance = Tolerance::new(1e-6, 1e-12);

struct ArcFit<'a, A: Atmosphere> {
    dynamics: &'a Dynamics<A>,
    observations: &'a [(f64, [f64; 3])],
}

impl<A: Atmosphere> ResidualModel<7> for ArcFit<'_, A> {
    fn residuals(&self, x: &[f64; 7], emit: &mut dyn FnMut(&[f64; 7], f64, f64)) {
        let at = propagate(self.dynamics, x, FIT_TOLERANCE);
        let w = 1.0 / (SIGMA * SIGMA);
        for (t, p) in self.observations {
            let (s, phi) = split::<7, 56>(&at(*t));
            for i in 0..3 {
                emit(&phi[i], s[i] - p[i], w);
            }
        }
    }
}

/// `SplitMix64` through Box–Muller: deterministic noise with no dependency.
struct Gaussian(u64);

impl Gaussian {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn next(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

struct Scenario {
    name: &'static str,
    altitude_km: f64,
    f107: f64,
    ap: f64,
}

struct Fit {
    x: [f64; 7],
    sigma: [f64; 7],
    /// `cost/(m − 7)`, the factor between `sigma` and the noise-only formal sigma.
    variance_factor: f64,
    iterations: u32,
    converged: bool,
    rms: f64,
}

fn fit<A: Atmosphere>(d: &Dynamics<A>, observations: &[(f64, [f64; 3])], x0: &[f64; 7]) -> Fit {
    let solver = GaussNewton {
        max_iterations: 30,
        ..GaussNewton::default()
    };
    let (x, covariance, report) = solver
        .solve(
            &ArcFit {
                dynamics: d,
                observations,
            },
            x0,
        )
        .expect("the arc is observable");
    let m = (3 * observations.len()) as f64;
    let variance_factor = report.cost / (m - 7.0);
    let mut sigma = [0.0; 7];
    for i in 0..7 {
        sigma[i] = (covariance[i][i] * variance_factor).sqrt();
    }
    Fit {
        x,
        sigma,
        variance_factor,
        iterations: report.iterations,
        converged: report.converged,
        rms: SIGMA * (report.cost / m).sqrt(),
    }
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The instrument, and the size of what the shipped lift drops, along the true arc.
fn along_the_arc<G: Fn(f64, [f64; 3]) -> (f64, f64, f64)>(
    field: &SdtmAtmosphere<G, 9>,
    drivers: &SdtmDrivers,
    truth: &impl Fn(f64) -> [f64; 56],
) {
    let (mut worst_normal, mut worst_value) = (0.0_f64, 0.0_f64);
    let mut ratios = Vec::new();
    for k in (0..OBSERVATIONS).step_by(2) {
        let t = k as f64 * CADENCE;
        let y = truth(t);
        let r = [y[0], y[1], y[2]];
        let g = gradient(field, t, r);
        let nh = normal(t, r);
        let gn = dot(g, nh);
        let gh = [g[0] - gn * nh[0], g[1] - gn * nh[1], g[2] - gn * nh[2]];
        ratios.push(norm(gh) / gn.abs());

        let (alt, lat, lon) = geodetic(t, r);
        let total = SEC0 + t;
        let sec = total - 86_400.0 * (total / 86_400.0).floor();
        let mut lst = sec / 3_600.0 + lon / 15.0;
        lst -= 24.0 * (lst / 24.0).floor();
        let point = SdtmPoint {
            altitude: alt * 1_000.0,
            latitude: lat.to_radians(),
            local_solar_time: lst,
            day_of_year: f64::from(DOY0) + total / 86_400.0,
        };
        let (rho, drho_dz) = field.field.density_gradient(&point, drivers);
        worst_value = worst_value.max((rho - field.density(t, &r)).abs() / rho);
        worst_normal = worst_normal.max((gn - drho_dz).abs() / drho_dz.abs());
    }
    ratios.sort_by(f64::total_cmp);
    let median = ratios[ratios.len() / 2];
    let max = ratios[ratios.len() - 1];

    println!(
        "instrument: value vs field {worst_value:.1e}, ∇ρ·n̂ vs analytic ∂ρ/∂z {worst_normal:.1e} (worst relative)"
    );
    println!(
        "|∇ₕρ|/|∂ρ/∂z| along the arc: median {median:.3e}, max {max:.3e}, {} samples",
        ratios.len()
    );
}

/// What the Jacobian loses after 24 h, at the true state.
fn jacobian_gap<A: Atmosphere, B: Atmosphere>(
    radial: &Dynamics<A>,
    full: &Dynamics<B>,
    x_true: &[f64; 7],
) {
    let (end_r, end_f) = (
        propagate(radial, x_true, FIT_TOLERANCE)(SPAN),
        propagate(full, x_true, FIT_TOLERANCE)(SPAN),
    );
    let (state_r, phi_r) = split::<7, 56>(&end_r);
    let (state_f, phi_f) = split::<7, 56>(&end_f);
    let state_gap = norm([
        state_r[0] - state_f[0],
        state_r[1] - state_f[1],
        state_r[2] - state_f[2],
    ]);
    let (mut diff, mut size) = (0.0, 0.0);
    for i in 0..3 {
        for j in 0..7 {
            diff += (phi_r[i][j] - phi_f[i][j]).powi(2);
            size += phi_f[i][j].powi(2);
        }
    }
    let stm_gap = (diff / size).sqrt();
    let drag_column = norm([
        phi_r[0][6] - phi_f[0][6],
        phi_r[1][6] - phi_f[1][6],
        phi_r[2][6] - phi_f[2][6],
    ]) / norm([phi_f[0][6], phi_f[1][6], phi_f[2][6]]);

    println!(
        "after 24 h at the true state: state gap {state_gap:.2e} m, position rows of Φ {stm_gap:.2e}, drag column {drag_column:.2e} (relative)"
    );
}

/// Returns whether any parameter moved beyond its interval.
fn run(s: &Scenario) -> bool {
    let drivers = SdtmDrivers {
        f107: s.f107,
        f107a: s.f107,
        storm: s.ap,
    };
    let sdtm = || SdtmAtmosphere {
        field: coefficients::FIELD,
        drivers,
        doy0: f64::from(DOY0),
        sec0: SEC0,
        geodetic,
    };
    let truth_model = dynamics(Nrlmsise00Atmosphere {
        doy0: DOY0,
        sec0: SEC0,
        f107: s.f107,
        f107a: s.f107,
        ap: s.ap,
        geodetic,
    });
    let radial = dynamics(sdtm());
    let full = dynamics(FullGradient(sdtm()));

    let a = RE + s.altitude_km * 1_000.0;
    let speed = (MU / a).sqrt();
    let (si, ci) = INCLINATION_DEG.to_radians().sin_cos();
    let x_true = [a, 0.0, 0.0, 0.0, speed * ci, speed * si, 1.0];

    println!("## {}", s.name);
    let truth = propagate(&truth_model, &x_true, Tolerance::new(1e-9, 1e-13));
    let mut noise = Gaussian(0x5D7A_0001);
    let observations: Vec<(f64, [f64; 3])> = (0..OBSERVATIONS)
        .map(|k| {
            let t = k as f64 * CADENCE;
            let y = truth(t);
            (
                t,
                [
                    y[0] + SIGMA * noise.next(),
                    y[1] + SIGMA * noise.next(),
                    y[2] + SIGMA * noise.next(),
                ],
            )
        })
        .collect();

    along_the_arc(&sdtm(), &drivers, &truth);
    jacobian_gap(&radial, &full, &x_true);

    let fr = fit(&radial, &observations, &x_true);
    let ff = fit(&full, &observations, &x_true);
    let mut moved = [0.0; 7];
    for i in 0..7 {
        moved[i] = (fr.x[i] - ff.x[i]).abs() / ff.sigma[i];
    }
    let (worst_i, worst) = moved
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("seven parameters");
    let names = ["x", "y", "z", "vx", "vy", "vz", "drag scale"];

    for (label, f) in [("radial", &fr), ("full", &ff)] {
        println!(
            "{label:>6}: {} iterations (converged {}), post-fit RMS {:.3} m, drag scale {:.6} ± {:.6}, position σ {:.3} m",
            f.iterations, f.converged, f.rms, f.x[6], f.sigma[6], f.sigma[0]
        );
    }
    print!("|radial − full|/σ:");
    for i in 0..7 {
        print!(" {} {:.2e}", names[i], moved[i]);
    }
    let position = (0..3)
        .map(|i| (fr.x[i] - ff.x[i]).abs())
        .fold(0.0, f64::max);
    let velocity = (3..6)
        .map(|i| (fr.x[i] - ff.x[i]).abs())
        .fold(0.0, f64::max);
    println!(
        "\nlargest: {} at {worst:.2e} σ; largest move {position:.2e} m, {velocity:.2e} m/s, drag scale {:.2e}",
        names[worst_i],
        (fr.x[6] - ff.x[6]).abs()
    );
    println!(
        "against the noise-only formal σ, which leaves out the model error: largest {} at {:.2e}\n",
        names[worst_i],
        worst * ff.variance_factor.sqrt()
    );
    worst > 1.0
}

fn main() {
    let scenarios = [
        Scenario {
            name: "250 km, F10.7 150, ap 15",
            altitude_km: 250.0,
            f107: 150.0,
            ap: 15.0,
        },
        Scenario {
            name: "450 km, F10.7 150, ap 15",
            altitude_km: 450.0,
            f107: 150.0,
            ap: 15.0,
        },
        Scenario {
            name: "450 km, F10.7 150, ap 100",
            altitude_km: 450.0,
            f107: 150.0,
            ap: 100.0,
        },
    ];
    let mut go = false;
    for s in &scenarios {
        go |= run(s);
    }
    println!(
        "verdict: {}",
        if go {
            "GO — some parameter moved beyond its interval"
        } else {
            "NO-GO — no parameter moved beyond its interval"
        }
    );
}
