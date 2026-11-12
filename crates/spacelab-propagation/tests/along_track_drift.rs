//! The gate of `plans/17` Ф30.H (F64): what a density error does to a predicted orbit.
//!
//! A density error `ε` on a drag acceleration `a_d` moves the prediction along track by
//! `1.5·ε·a_d·t²`. `derivations/along_track_drift.mac` derives that from Hill–Clohessy–Wiltshire:
//! the along-track response to an in-plane acceleration `(f_x, f_y)` is the convolution
//! `y(t) = ∫₀ᵗ [G_yx(t−s) f_x(s) + G_yy(t−s) f_y(s)] ds`, with `G_yy(τ) = −3τ + (4/n) sin nτ` and
//! `G_yx(τ) = (2/n)(cos nτ − 1)`; a constant `f_y` gives `−(3/2) f_y t²` plus a remainder at most
//! `(16/3)/(n t)²` of it.
//!
//! Here two propagations start from one state, under SDTM-1's density `ρ` and under `(1 + ε)ρ`, with
//! point mass, `J₂` and co-rotating drag. Their along-track separation is held to two predictions:
//!
//! - the HCW convolution of the drag-error acceleration `ε·a_drag`, sampled on the reference orbit
//!   every 20 s, with the drag-density feedback of the derivation's section 6, within **1 %**;
//! - F64's `1.5·ε·ā_d·t²`, with `ā_d` the mean along-track drag acceleration over `[0, t]`, within
//!   **5 %**: the bounded remainder adds 0.9 % at 6 h, and the day–night modulation of `a_d` resonates
//!   with the orbit and adds a term that falls as `1/(n t)`.
//!
//! Both bounds were fixed before the first run. That run held the convolution *without* the feedback
//! to 1 % and failed it: 1.19 % at 350 km after 72 h. The perturbed orbit flies lower, through denser
//! air, and that term is of first order in `ε`, not second as first assumed; its share,
//! `(2/3)·k·t/n`, is 1.0 % there and 0.17 % at 450 km. Section 6 derives it and the convolution now
//! carries it; the bound is unchanged. Both convolutions are printed.

#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    // `t`, `r`, `n`, `a` and the axes are the symbols of the equations this checks.
    clippy::many_single_char_names
)]

use spacelab_autodiff::Scalar;
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::forces::{
    Acceleration, Atmosphere, Drag, J2, Newtonian, NoWind, Plain, PointMass,
};
use spacelab_propagation::sdtm::{SdtmAtmosphere, SdtmDrivers, coefficients};
use spacelab_propagation::{Dop853, Tolerance};

const MU: f64 = 3.986_004_415e14;
const RE: f64 = 6_378_136.3;
const C20_BAR: f64 = -0.484_165_143_790_815e-3;
const OMEGA: f64 = 7.292_115e-5;
const BALLISTIC: f64 = 0.011;
const SPAN: f64 = 72.0 * 3_600.0;
const SAMPLE: f64 = 20.0;
/// `3600 / SAMPLE`.
const SAMPLES_PER_HOUR: usize = 180;
/// Radial step of the feedback gain's central difference, m.
const STEP: f64 = 10.0;
const HORIZONS_H: [usize; 5] = [6, 12, 24, 48, 72];

/// Spherical altitude and geocentric latitude under a uniformly rotating Earth: enough for the
/// density to vary along the orbit as the real one does, by height, latitude and local time.
fn geodetic(t: f64, r: [f64; 3]) -> (f64, f64, f64) {
    let norm = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    let (s, c) = (OMEGA * t).sin_cos();
    let (x, y) = (c * r[0] + s * r[1], -s * r[0] + c * r[1]);
    (
        (norm - 6_378_137.0) / 1_000.0,
        (r[2] / norm).asin().to_degrees(),
        y.atan2(x).to_degrees(),
    )
}

struct Scaled<A>(A, f64);

impl<A: Atmosphere> Atmosphere for Scaled<A> {
    fn density<S: Scalar>(&self, t: f64, r: &[S; 3]) -> S {
        S::from_f64(self.1) * self.0.density(t, r)
    }
}

type Geodetic = fn(f64, [f64; 3]) -> (f64, f64, f64);

fn sdtm() -> SdtmAtmosphere<Geodetic, 9> {
    SdtmAtmosphere {
        field: coefficients::FIELD,
        drivers: SdtmDrivers {
            f107: 150.0,
            f107a: 150.0,
            storm: 15.0,
        },
        doy0: 80.0,
        sec0: 43_200.0,
        geodetic,
    }
}

fn drag<A: Atmosphere>(atmosphere: A) -> Drag<A, NoWind> {
    Drag {
        ballistic_coefficient: BALLISTIC,
        earth_rotation_rate: OMEGA,
        atmosphere,
        wind: NoWind,
    }
}

fn gravity() -> (PointMass, J2) {
    (
        PointMass { mu: MU },
        J2 {
            mu: MU,
            re: RE,
            j2: -C20_BAR * 5.0_f64.sqrt(),
        },
    )
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
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

/// Radial and along-track unit vectors of a state.
fn axes(y: &[f64; 6]) -> ([f64; 3], [f64; 3]) {
    let r = [y[0], y[1], y[2]];
    let radial = unit(r);
    let normal = unit(cross(r, [y[3], y[4], y[5]]));
    (radial, cross(normal, radial))
}

/// The radial offset of the first-order solution, `G_xx = sin nτ / n` and `G_xy = 2(1 − cos nτ)/n`,
/// which the feedback multiplies. Both kernels are rotations, so the convolution is a running sum
/// of the forcing times `e^{−i n s}`, turned by `e^{i n t}`; the trapezoid halves the two end terms.
fn radial_offset(fx: &[f64], fy: &[f64], n: f64) -> Vec<f64> {
    let rotated = |i: usize| {
        let (sn, cs) = (n * i as f64 * SAMPLE).sin_cos();
        ([fx[i] * cs, -fx[i] * sn], [fy[i] * cs, -fy[i] * sn])
    };
    let (p_first, q_first) = rotated(0);
    let (mut p_sum, mut q_sum, mut fy_sum) = ([0.0; 2], [0.0; 2], 0.0);
    let mut x0 = Vec::with_capacity(fx.len());
    for j in 0..fx.len() {
        let (p_j, q_j) = rotated(j);
        p_sum = [p_sum[0] + p_j[0], p_sum[1] + p_j[1]];
        q_sum = [q_sum[0] + q_j[0], q_sum[1] + q_j[1]];
        fy_sum += fy[j];
        let p = [
            p_sum[0] - 0.5 * (p_first[0] + p_j[0]),
            p_sum[1] - 0.5 * (p_first[1] + p_j[1]),
        ];
        let q = [
            q_sum[0] - 0.5 * (q_first[0] + q_j[0]),
            q_sum[1] - 0.5 * (q_first[1] + q_j[1]),
        ];
        let total = fy_sum - 0.5 * (fy[0] + fy[j]);
        let (sn, cs) = (n * j as f64 * SAMPLE).sin_cos();
        let sine = cs * p[1] + sn * p[0];
        let cosine = cs * q[0] - sn * q[1];
        x0.push(SAMPLE / n * (sine + 2.0 * (total - cosine)));
    }
    x0
}

struct Separation {
    hours: usize,
    propagated: f64,
    hcw: f64,
    with_feedback: f64,
    formula: f64,
}

/// Per horizon: the propagated along-track separation and the three predictions of it.
fn separations(altitude_km: f64, eps: f64) -> Vec<Separation> {
    let a = RE + altitude_km * 1_000.0;
    let speed = (MU / a).sqrt();
    let (si, ci) = 51.6_f64.to_radians().sin_cos();
    let y0 = [a, 0.0, 0.0, 0.0, speed * ci, speed * si];
    let controls = Controls::new(Tolerance::new(1e-9, 1e-13), 1e-3);
    let reference = integrate(
        &Dop853,
        &Plain(Newtonian((gravity(), drag(sdtm())))),
        0.0,
        &y0,
        SPAN,
        30.0,
        controls,
    )
    .expect("the reference integrates");
    let perturbed = integrate(
        &Dop853,
        &Plain(Newtonian((gravity(), drag(Scaled(sdtm(), 1.0 + eps))))),
        0.0,
        &y0,
        SPAN,
        30.0,
        controls,
    )
    .expect("the perturbed orbit integrates");

    // The drag-error acceleration on the reference orbit, in its radial and along-track axes.
    let force = drag(sdtm());
    let n_samples = 72 * SAMPLES_PER_HOUR;
    let mut fx = Vec::with_capacity(n_samples + 1);
    let mut fy = Vec::with_capacity(n_samples + 1);
    let mut gain = Vec::with_capacity(n_samples + 1);
    for k in 0..=n_samples {
        let t = k as f64 * SAMPLE;
        let y = reference.eval(t);
        let v = [y[3], y[4], y[5]];
        let acc = force.acceleration(t, &[y[0], y[1], y[2]], &v);
        let (radial, along) = axes(&y);
        fx.push(eps * dot(acc, radial));
        fy.push(eps * dot(acc, along));
        let at = |h: f64| {
            let r = [
                y[0] + h * radial[0],
                y[1] + h * radial[1],
                y[2] + h * radial[2],
            ];
            dot(force.acceleration(t, &r, &v), along)
        };
        gain.push((at(STEP) - at(-STEP)) / (2.0 * STEP));
    }
    let n = (MU / (a * a * a)).sqrt();
    // The kernels depend on the lag alone, so tabulate them once per sample lag.
    let (sin_lag, cos_lag): (Vec<f64>, Vec<f64>) = (0..=n_samples)
        .map(|k| (n * k as f64 * SAMPLE).sin_cos())
        .unzip();
    let weight = |k: usize, last: usize| if k == 0 || k == last { 0.5 } else { 1.0 };
    let x0 = radial_offset(&fx, &fy, n);
    HORIZONS_H
        .iter()
        .map(|&hours| {
            let last = hours * SAMPLES_PER_HOUR;
            let t = last as f64 * SAMPLE;
            let yr = reference.eval(t);
            let yp = perturbed.eval(t);
            let (_, along) = axes(&yr);
            let propagated = dot(sub([yp[0], yp[1], yp[2]], [yr[0], yr[1], yr[2]]), along);
            // Trapezoid over [0, t] of G(t − s)·f(s).
            let (mut hcw, mut feedback, mut mean) = (0.0, 0.0, 0.0);
            for k in 0..=last {
                let w = weight(k, last) * SAMPLE;
                let lag = last - k;
                let tau = lag as f64 * SAMPLE;
                let along_kernel = -3.0 * tau + 4.0 / n * sin_lag[lag];
                let radial_kernel = 2.0 / n * (cos_lag[lag] - 1.0);
                hcw += w * (along_kernel * fy[k] + radial_kernel * fx[k]);
                feedback += w * along_kernel * gain[k] * x0[k];
                mean += w * fy[k];
            }
            mean /= t;
            Separation {
                hours,
                propagated,
                hcw,
                with_feedback: hcw + feedback,
                formula: -1.5 * mean * t * t,
            }
        })
        .collect()
}

#[test]
fn a_density_error_moves_the_prediction_along_track_as_hill_clohessy_wiltshire_says() {
    let (mut worst_plain, mut worst_hcw, mut worst_formula) = (0.0_f64, 0.0_f64, 0.0_f64);
    for altitude_km in [350.0, 450.0] {
        for eps in [0.2, -0.2] {
            for s in separations(altitude_km, eps) {
                let plain_error = s.propagated / s.hcw - 1.0;
                let feedback_error = s.propagated / s.with_feedback - 1.0;
                let formula_error = s.propagated / s.formula - 1.0;
                println!(
                    "{altitude_km} km, eps {eps:+}, {:>2} h: along-track {:+.3e} m, HCW {plain_error:+.2e}, \
                     HCW with feedback {feedback_error:+.2e}, 1.5 eps a_d t^2 {formula_error:+.2e}",
                    s.hours, s.propagated
                );
                worst_plain = worst_plain.max(plain_error.abs());
                worst_hcw = worst_hcw.max(feedback_error.abs());
                worst_formula = worst_formula.max(formula_error.abs());
            }
        }
    }
    println!(
        "worst: HCW {worst_plain:.2e}, HCW with feedback {worst_hcw:.2e}, formula {worst_formula:.2e}"
    );
    assert!(worst_hcw < 0.01, "HCW convolution off by {worst_hcw:.2e}");
    assert!(
        worst_formula < 0.05,
        "F64's formula off by {worst_formula:.2e}"
    );
}
