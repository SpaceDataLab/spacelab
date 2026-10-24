//! The DSST short-period functions of the conservative forces against the equation that defines
//! them.
//!
//! A first-order short-period function η(λ) of an element E is the periodic solution of
//! `n·∂η/∂λ = F(λ) − ⟨F⟩`, where `F` is the element's instantaneous rate under the perturbation on
//! the Keplerian orbit of the mean elements and `⟨F⟩` its mean over the mean longitude λ. The mean
//! longitude also moves with the semi-major axis, so its equation carries `+ (3/2)(n/a)·η_a` on the
//! left; a tesseral term also turns with the central body, which adds `θ̇·∂η/∂θ`. Both sides are
//! linear in the perturbation at fixed mean elements, so the residual measures only where the
//! shipped series are cut — not a second-order effect. The bounds below sit above the measured
//! residual of each cut, and the third-body test shows its residual falling as (a/R₃)³.
//!
//! Nothing here goes through the DSST machinery but the function under test. `F` is Gauss's
//! `∂E/∂v · a_pert`: the perturbing acceleration of a force model outside `dsst`, and the velocity
//! partials by central differences of a Cartesian → equinoctial conversion written in this file.
//! `∂η/∂λ` and `∂η/∂θ` are spectral derivatives of η sampled on a uniform grid.
//!
//! The same `F` also defines the mean rates the Gaussian (drag, SRP) contributions integrate:
//! `⟨F⟩` over an arc of the orbit is `(1/2π)∫F dλ`, which the last two tests take by Simpson's rule
//! in the mean longitude while the shipped code takes Gauss–Legendre in the true longitude.

#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

use core::f64::consts::TAU;

use spacelab_propagation::dsst::gaussian::{NodeState, mean_element_rates, variational_matrix};
use spacelab_propagation::dsst::{DsstAuxiliary, DsstTesseral, DsstThirdBody, DsstZonal};
use spacelab_propagation::geopotential::GeoPotential;

const MU: f64 = 3.986_004_415e14;
const RE: f64 = 6_378_136.3;
/// Samples of the mean longitude: the spectral derivative is exact far below every residual here.
const N: usize = 128;
const LEO: [f64; 5] = [7.0e6, 0.008, -0.006, 0.3, 0.35];

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

fn comb(x: f64, u: [f64; 3], y: f64, w: [f64; 3]) -> [f64; 3] {
    [
        x * u[0] + y * w[0],
        x * u[1] + y * w[1],
        x * u[2] + y * w[2],
    ]
}

/// The equinoctial frame's f and g for retrograde factor +1 (Orekit's `hx = q`, `hy = p`).
fn frame(q: f64, p: f64) -> ([f64; 3], [f64; 3]) {
    let c = 1.0 + p * p + q * q;
    (
        [(1.0 - p * p + q * q) / c, 2.0 * p * q / c, -2.0 * p / c],
        [2.0 * p * q / c, (1.0 + p * p - q * q) / c, 2.0 * q / c],
    )
}

/// `[a, k, h, q, p, λ]` → position and velocity, through the eccentric longitude.
fn to_cartesian(e: [f64; 6]) -> ([f64; 3], [f64; 3]) {
    let [a, k, h, q, p, lam] = e;
    let mut ff = lam;
    for _ in 0..50 {
        let res = ff - k * ff.sin() + h * ff.cos() - lam;
        ff -= res / (1.0 - k * ff.cos() - h * ff.sin());
    }
    let b = 1.0 / (1.0 + (1.0 - k * k - h * h).sqrt());
    let (sf, cf) = ff.sin_cos();
    let x = a * ((1.0 - h * h * b) * cf + h * k * b * sf - k);
    let y = a * ((1.0 - k * k * b) * sf + h * k * b * cf - h);
    let r = a * (1.0 - k * cf - h * sf);
    let vscale = a * a * (MU / (a * a * a)).sqrt() / r;
    let xd = vscale * (h * k * b * cf - (1.0 - h * h * b) * sf);
    let yd = vscale * ((1.0 - k * k * b) * cf - h * k * b * sf);
    let (f, g) = frame(q, p);
    (comb(x, f, y, g), comb(xd, f, yd, g))
}

/// Position and velocity → `[a, k, h, q, p, λ]`, through the angular-momentum and eccentricity
/// vectors and the true longitude.
fn to_equinoctial(r: [f64; 3], v: [f64; 3]) -> [f64; 6] {
    let hv = cross(r, v);
    let hn = dot(hv, hv).sqrt();
    let w = [hv[0] / hn, hv[1] / hn, hv[2] / hn];
    let q = -w[1] / (1.0 + w[2]);
    let p = w[0] / (1.0 + w[2]);
    let rn = dot(r, r).sqrt();
    let a = 1.0 / (2.0 / rn - dot(v, v) / MU);
    let vh = cross(v, hv);
    let ev = [
        vh[0] / MU - r[0] / rn,
        vh[1] / MU - r[1] / rn,
        vh[2] / MU - r[2] / rn,
    ];
    let (f, g) = frame(q, p);
    let k = dot(ev, f);
    let h = dot(ev, g);
    let lv = dot(r, g).atan2(dot(r, f));
    [a, k, h, q, p, mean_longitude(k, h, lv)]
}

/// True longitude → mean longitude, through the eccentric longitude.
fn mean_longitude(k: f64, h: f64, lv: f64) -> f64 {
    let (sl, cl) = lv.sin_cos();
    let le = lv
        + 2.0 * ((h * cl - k * sl) / (1.0 + (1.0 - k * k - h * h).sqrt() + k * cl + h * sl)).atan();
    le - k * le.sin() + h * le.cos()
}

fn wrap(x: f64) -> f64 {
    x - TAU * (x / TAU).round()
}

/// Gauss's rates `∂E/∂v · acc` at the Keplerian state of `e`, the velocity partials by central
/// differences.
fn gauss_rates(e: [f64; 6], acc: &impl Fn([f64; 3], [f64; 3]) -> [f64; 3]) -> [f64; 6] {
    let (r, v) = to_cartesian(e);
    let acc = acc(r, v);
    let dv = 0.05;
    let mut rate = [0.0; 6];
    for (axis, &ai) in acc.iter().enumerate() {
        let mut vp = v;
        let mut vm = v;
        vp[axis] += dv;
        vm[axis] -= dv;
        let ep = to_equinoctial(r, vp);
        let em = to_equinoctial(r, vm);
        for (i, ri) in rate.iter_mut().enumerate() {
            let d = if i == 5 {
                wrap(ep[i] - em[i])
            } else {
                ep[i] - em[i]
            };
            *ri += d / (2.0 * dv) * ai;
        }
    }
    rate
}

/// The spectral derivative of a periodic sample on the uniform grid `2πk/len`.
fn spectral_derivative(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![0.0; n];
    for j in 1..n / 2 {
        let (mut re, mut im) = (0.0, 0.0);
        for (k, &yk) in y.iter().enumerate() {
            let t = TAU * (j * k) as f64 / n as f64;
            re += yk * t.cos();
            im -= yk * t.sin();
        }
        re *= 2.0 / n as f64;
        im *= 2.0 / n as f64;
        // y ∋ re cos jλ − im sin jλ; its derivative is −j re sin jλ − j im cos jλ.
        for (k, o) in out.iter_mut().enumerate() {
            let t = TAU * (j * k) as f64 / n as f64;
            *o -= j as f64 * (re * t.sin() + im * t.cos());
        }
    }
    out
}

/// One element's verdict: the worst residual of the defining equation relative to the worst
/// `|F − ⟨F⟩|`, and the mean of η relative to its amplitude.
#[derive(Debug, Clone, Copy)]
struct Verdict {
    residual: f64,
    mean: f64,
}

/// The six verdicts over the grid. `eta(λ, θ)` and `acc(θ, r, v)` take the central-body angle θ,
/// which only the tesseral contribution reads; `nt` samples it over a turn and adds `θ̇·∂η/∂θ`.
fn verdicts(
    mean: [f64; 5],
    nt: usize,
    theta_dot: f64,
    eta: impl Fn(f64, f64) -> [f64; 6],
    acc: impl Fn(f64, [f64; 3], [f64; 3]) -> [f64; 3],
) -> [Verdict; 6] {
    let [a, k, h, q, p] = mean;
    let n = (MU / (a * a * a)).sqrt();
    let grid = |m: usize| -> Vec<f64> { (0..m).map(|i| TAU * i as f64 / m as f64).collect() };
    // Row-major [θ][λ].
    let mut etas = Vec::with_capacity(N * nt);
    let mut rates = Vec::with_capacity(N * nt);
    for &t in &grid(nt) {
        for &l in &grid(N) {
            etas.push(eta(l, t));
            rates.push(gauss_rates([a, k, h, q, p, l], &|r, v| acc(t, r, v)));
        }
    }
    core::array::from_fn(|i| {
        let y: Vec<f64> = etas.iter().map(|e| e[i]).collect();
        let f: Vec<f64> = rates.iter().map(|r| r[i]).collect();
        let mut lhs = vec![0.0; N * nt];
        for (row, out) in y.chunks(N).zip(lhs.chunks_mut(N)) {
            for (o, d) in out.iter_mut().zip(spectral_derivative(row)) {
                *o += n * d;
            }
        }
        if nt > 1 {
            for c in 0..N {
                let col: Vec<f64> = (0..nt).map(|b| y[b * N + c]).collect();
                for (b, d) in spectral_derivative(&col).into_iter().enumerate() {
                    lhs[b * N + c] += theta_dot * d;
                }
            }
        }
        let fbar = f.iter().sum::<f64>() / f.len() as f64;
        let ybar = y.iter().sum::<f64>() / y.len() as f64;
        let (mut worst, mut scale, mut amp) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (kk, (&l, &fk)) in lhs.iter().zip(&f).enumerate() {
            let coupling = if i == 5 {
                1.5 * n / a * etas[kk][0]
            } else {
                0.0
            };
            worst = worst.max((l + coupling - (fk - fbar)).abs());
            scale = scale.max((fk - fbar).abs());
            amp = amp.max((y[kk] - ybar).abs());
        }
        Verdict {
            residual: worst / scale,
            mean: ybar.abs() / amp,
        }
    })
}

fn assert_verdicts(label: &str, v: &[Verdict; 6], bound: f64) {
    for (i, e) in v.iter().enumerate() {
        assert!(
            e.residual < bound,
            "{label}, element {i}: residual {:.3e} ≥ {bound:.0e}",
            e.residual
        );
        assert!(
            e.mean < 1e-12,
            "{label}, element {i}: mean {:.3e} of the amplitude",
            e.mean
        );
    }
}

fn aux(mean: [f64; 5], lam: f64) -> DsstAuxiliary {
    DsstAuxiliary::new(mean[0], mean[1], mean[2], mean[3], mean[4], lam, MU)
}

/// The perturbing part of a field's acceleration: the total less the point mass it includes.
fn perturbing(g: [f64; 3], r: [f64; 3]) -> [f64; 3] {
    let r3 = dot(r, r).powf(1.5);
    [
        g[0] + MU * r[0] / r3,
        g[1] + MU * r[1] / r3,
        g[2] + MU * r[2] / r3,
    ]
}

/// The third-body tide on a geocentric satellite from a body fixed at `s`.
fn tide(mu3: f64, s: [f64; 3], r: [f64; 3]) -> [f64; 3] {
    let d = [s[0] - r[0], s[1] - r[1], s[2] - r[2]];
    let d3 = dot(d, d).powf(1.5);
    let s3 = dot(s, s).powf(1.5);
    [
        mu3 * (d[0] / d3 - s[0] / s3),
        mu3 * (d[1] / d3 - s[1] / s3),
        mu3 * (d[2] / d3 - s[2] / s3),
    ]
}

#[test]
fn the_round_trip_is_the_identity() {
    let e = [7.0e6, 0.008, -0.006, 0.3, 0.35, 2.0];
    let (r, v) = to_cartesian(e);
    let back = to_equinoctial(r, v);
    for (b, x) in back.iter().zip(e) {
        assert!((b - x).abs() < 1e-12 * x.abs().max(1.0), "{b} vs {x}");
    }
}

/// J2..J6 with Orekit's default cuts: `s ≤ min(4, n − 1)`, `j ≤ 2n + 1`. Measured 1.6e-10 at
/// e = 0.01 and 2.1e-7 at e = 0.05, the size of the e⁵ the s-cut drops.
#[test]
fn zonal_short_periods_solve_their_defining_equation() {
    let j = [
        1.082_626_7e-3,
        -2.532_4e-6,
        -1.619_6e-6,
        -2.272_9e-7,
        5.406_8e-7,
    ];
    let zonal = DsstZonal::from_j(MU, RE, &j);
    let coeffs: Vec<(usize, usize, f64, f64)> = j
        .iter()
        .enumerate()
        .map(|(i, &jn)| (i + 2, 0, -jn, 0.0))
        .collect();
    let geo = GeoPotential::from_unnormalized(MU, RE, 6, &coeffs).expect("degree 6 is supported");
    for (mean, bound) in [(LEO, 1e-9), ([2.6e7, 0.03, 0.04, -0.2, 0.1], 1e-6)] {
        let v = verdicts(
            mean,
            1,
            0.0,
            |l, _| zonal.short_period_correction(&aux(mean, l)),
            |_, r, _| perturbing(geo.acceleration_simd(&r), r),
        );
        assert_verdicts("zonal", &v, bound);
    }
}

/// The Moon held fixed, as the DSST holds it over one revolution. Measured 2.4e-5 in LEO and
/// 1.8e-3 in GEO, where a/R₃ is 0.11.
#[test]
fn third_body_short_periods_solve_their_defining_equation() {
    let mu3 = 4.902_800_066e12;
    let moon = DsstThirdBody::new(MU, mu3);
    let s = [2.1e8, -3.0e8, 1.1e8];
    for (mean, bound) in [(LEO, 1e-4), ([4.2164e7, 0.02, -0.01, 0.05, -0.03], 5e-3)] {
        let v = verdicts(
            mean,
            1,
            0.0,
            |l, _| moon.short_period_correction(&aux(mean, l), s),
            |_, r, _| tide(mu3, s, r),
        );
        assert_verdicts("third body", &v, bound);
    }
}

/// The GEO residual is the a/R₃ cut and not an error: with the tide µ₃/R₃³ held fixed, each halving
/// of a/R₃ divides it by about 8 (measured 7.5, 7.5, 8.0).
#[test]
fn the_third_body_residual_is_the_distance_cut() {
    let mean = [4.2164e7, 0.02, -0.01, 0.05, -0.03];
    let worst: Vec<f64> = [1.0_f64, 2.0, 4.0, 8.0]
        .iter()
        .map(|&scale| {
            let mu3 = 4.902_800_066e12 * scale.powi(3);
            let moon = DsstThirdBody::new(MU, mu3);
            let s = [2.1e8 * scale, -3.0e8 * scale, 1.1e8 * scale];
            let v = verdicts(
                mean,
                1,
                0.0,
                |l, _| moon.short_period_correction(&aux(mean, l), s),
                |_, r, _| tide(mu3, s, r),
            );
            v.iter().map(|e| e.residual).fold(0.0, f64::max)
        })
        .collect();
    for pair in worst.windows(2) {
        assert!(
            pair[0] / pair[1] > 6.0,
            "halving a/R₃ took {:.3e} to {:.3e}",
            pair[0],
            pair[1]
        );
    }
}

/// A degree-4 tesseral field on a non-resonant LEO, over the torus (λ, θ). Measured 1.2e-6, the
/// size of the e³ the m-daily s-cut `min(4, d − 2)` drops at e = 0.01.
#[test]
fn tesseral_short_periods_solve_their_defining_equation() {
    let theta_dot = 7.292_115e-5;
    // Unnormalised C_nm, S_nm, m ≥ 1: Earth-sized, signs as in EGM.
    let cs: [(usize, usize, f64, f64); 9] = [
        (2, 1, -2.4e-10, 1.5e-9),
        (2, 2, 1.574_5e-6, -9.039e-7),
        (3, 1, 2.191_9e-6, 2.688e-7),
        (3, 2, 3.089e-7, -2.115e-7),
        (3, 3, 1.006e-7, 1.972e-7),
        (4, 1, -5.087e-7, -4.494e-7),
        (4, 2, 7.783e-8, 1.481e-7),
        (4, 3, 5.92e-8, -1.201e-8),
        (4, 4, -3.98e-9, 6.52e-9),
    ];
    let mut cnm: Vec<Vec<f64>> = (0..5).map(|n| vec![0.0; n + 1]).collect();
    let mut snm = cnm.clone();
    for &(n, m, c, s) in &cs {
        cnm[n][m] = c;
        snm[n][m] = s;
    }
    let tess = DsstTesseral::new(MU, RE, theta_dot, cnm, snm);
    let geo = GeoPotential::from_unnormalized(MU, RE, 4, &cs).expect("degree 4 is supported");
    let v = verdicts(
        LEO,
        16,
        theta_dot,
        |l, t| tess.short_period_correction(&aux(LEO, l), t),
        |t, r, _| {
            // Body frame turned by θ about +Z: r_b = R(θ) r, a = R(θ)ᵀ a_b.
            let (st, ct) = t.sin_cos();
            let rb = [ct * r[0] + st * r[1], -st * r[0] + ct * r[1], r[2]];
            let ab = perturbing(geo.acceleration_simd(&rb), rb);
            [ct * ab[0] - st * ab[1], st * ab[0] + ct * ab[1], ab[2]]
        },
    );
    assert_verdicts("tesseral", &v, 5e-6);
}

/// A drag-like acceleration written here, not taken from `dsst::drag`: an exponential atmosphere
/// turning with the Earth, `−½ (C_D A/m) ρ |v_r| v_r`.
fn drag_like(r: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    let omega = 7.292_115e-5;
    let vr = [v[0] + omega * r[1], v[1] - omega * r[0], v[2]];
    let rho = 3.0e-12 * (-(dot(r, r).sqrt() - 7.0e6) / 7.0e4).exp();
    let c = -0.5 * 2.2 * 0.01 * rho * dot(vr, vr).sqrt();
    [c * vr[0], c * vr[1], c * vr[2]]
}

/// An SRP-like acceleration from a Sun fixed at one astronomical unit, `P (A/m) AU² (r − s)/|r − s|³`.
fn srp_like(r: [f64; 3], _: [f64; 3]) -> [f64; 3] {
    let au = 1.495_978_707e11;
    let s = [0.6 * au, -0.7 * au, 0.38 * au];
    let d = [r[0] - s[0], r[1] - s[1], r[2] - s[2]];
    let c = 4.56e-6 * 0.02 * au * au / dot(d, d).powf(1.5);
    [c * d[0], c * d[1], c * d[2]]
}

/// The worst relative difference over the six rates, each against the largest of the six scaled
/// alike: the semi-major axis by `1/a`, the rest as they are.
fn worst_rate_difference(a: f64, got: [f64; 6], want: [f64; 6]) -> f64 {
    let scaled = |x: [f64; 6]| -> [f64; 6] {
        core::array::from_fn(|i| if i == 0 { x[0] / a } else { x[i] })
    };
    let (g, w) = (scaled(got), scaled(want));
    let scale = w.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    g.iter()
        .zip(w)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

/// `variational_matrix` is the velocity Jacobian of the elements, row by row, at twelve true
/// longitudes on a near-circular LEO and on an inclined orbit with e = 0.25. Measured 1.1e-9 at
/// worst, in the semi-major-axis row, the differences' own (δv/v)² error.
#[test]
fn the_gauss_matrix_is_the_velocity_jacobian_of_the_elements() {
    for mean in [LEO, [2.6e7, 0.15, -0.2, 0.3, -0.25]] {
        let [a, k, h, q, p] = mean;
        let (f, g) = frame(q, p);
        for i in 0..12 {
            let lam = TAU * f64::from(i) / 12.0 + 0.1;
            let (r, v) = to_cartesian([a, k, h, q, p, lam]);
            let lv = dot(r, g).atan2(dot(r, f));
            let b = variational_matrix(&aux(mean, lam), lv);
            let dv = 0.05;
            let mut fd = [[0.0; 3]; 6];
            for axis in 0..3 {
                let (mut vp, mut vm) = (v, v);
                vp[axis] += dv;
                vm[axis] -= dv;
                let (ep, em) = (to_equinoctial(r, vp), to_equinoctial(r, vm));
                for (el, row) in fd.iter_mut().enumerate() {
                    let d = if el == 5 {
                        wrap(ep[el] - em[el])
                    } else {
                        ep[el] - em[el]
                    };
                    row[axis] = d / (2.0 * dv);
                }
            }
            for (el, (got, want)) in b.iter().zip(fd).enumerate() {
                let scale = want.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                let worst = got
                    .iter()
                    .zip(want)
                    .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
                assert!(
                    worst < 1e-8 * scale,
                    "a = {a}, λ = {lam}, element {el}: {got:?} vs {want:?}"
                );
            }
        }
    }
}

/// `mean_element_rates` over an arc [L₁, L₂] of true longitude is `(1/2π)∫F dλ` over the matching
/// arc of mean longitude: the drag-like force over the full turn of a LEO and over an arc of an
/// eccentric orbit, the SRP-like force over the full turn of an inclined MEO. Measured 2.2e-10 at
/// worst, the central differences' own error.
#[test]
fn gaussian_mean_rates_are_the_orbit_average_of_gauss_rates() {
    type Force = fn([f64; 3], [f64; 3]) -> [f64; 3];
    let full = |mean: [f64; 5]| -> [f64; 2] {
        let (r, _) = to_cartesian([mean[0], mean[1], mean[2], mean[3], mean[4], 0.5]);
        let (f, g) = frame(mean[3], mean[4]);
        let lv = dot(r, g).atan2(dot(r, f));
        [lv - core::f64::consts::PI, lv + core::f64::consts::PI]
    };
    let eccentric = [7.4e6, 0.06, -0.04, 0.2, 0.15];
    let meo = [2.6e7, 0.15, -0.2, 0.3, -0.25];
    let cases: [(&str, [f64; 5], [f64; 2], Force); 3] = [
        ("drag, full turn", LEO, full(LEO), drag_like),
        ("drag, arc", eccentric, [-0.9, 1.3], drag_like),
        ("srp, full turn", meo, full(meo), srp_like),
    ];
    for (label, mean, [l1, l2], force) in cases {
        let [a, k, h, q, p] = mean;
        let got = mean_element_rates(&aux(mean, 0.5), l1, l2, |node: &NodeState| {
            force(node.pos, node.vel)
        });
        // The arc in mean longitude: λ − L is periodic, so the arc's length is exact.
        let offset = |lv: f64| wrap(mean_longitude(k, h, lv) - lv);
        let m1 = l1 + offset(l1);
        let m2 = l2 + offset(l2);
        let panels = 4096;
        let step = (m2 - m1) / f64::from(panels);
        let mut want = [0.0; 6];
        for i in 0..=panels {
            let weight = if i == 0 || i == panels {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let rate = gauss_rates([a, k, h, q, p, m1 + step * f64::from(i)], &force);
            for (w, r) in want.iter_mut().zip(rate) {
                *w += weight * step / 3.0 * r / TAU;
            }
        }
        let worst = worst_rate_difference(a, got, want);
        assert!(worst < 1e-9, "{label}: {worst:.3e}, {got:?} vs {want:?}");
    }
}
