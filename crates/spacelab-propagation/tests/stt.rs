//! The second-order state-transition tensor of a two-body + J₂ orbit, checked against finite
//! differences — the honest oracle, computed in-test, with no external tool.
//!
//! Four checks:
//! * **STT vs a finite difference of the STM.** `Ψᵢⱼₖ = ∂Φᵢⱼ/∂x₀ₖ`, so a central difference of the
//!   first-order STM about `x₀` reproduces `Ψ` to the FD floor. Differencing `Φ` (already a first
//!   derivative) is far tighter than second-differencing the state.
//! * **The second-order deviation map beats the STM.** For a finite `δx₀`, the STM residual is
//!   `O(‖δx₀‖²)`; adding the `½ Ψ δx₀ δx₀` term drops it to `O(‖δx₀‖³)`. We propagate `x₀+δx₀`
//!   nonlinearly and show the STT prediction cuts the residual by more than an order of magnitude,
//!   and that halving `δx₀` roughly halves the improvement ratio (the `O(‖δx₀‖)` signature).
//! * **The second-order covariance matches Gauss–Hermite.** A 3-point-per-axis Gauss–Hermite rule is
//!   exact for the quartic moments of the quadratic map, so it is an independent oracle for the
//!   Park–Scheeres mean/covariance formulas — no Monte-Carlo noise, no circularity.
//! * **A linear force has zero STT.** With `∂²f/∂x² = 0` the coupling term vanishes and `Ψ` stays 0
//!   to round-off, isolating the `H:ΦΦ` term of the variational equation.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::type_complexity
)]

use spacelab_autodiff::Scalar;
use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::{J2, Newtonian, Plain, PointMass};
use spacelab_propagation::stt::{
    VariationalStt, augment_stt, augmented_len, map_covariance, map_deviation, split_stt,
};
use spacelab_propagation::variational::{ScalarSystem, Variational, augment, split};
use spacelab_propagation::{Dop853, Tolerance};
use spacelab_testkit::nan_max;

const MU: f64 = 3.986_004_418e14; // m³/s², EGM-96 Earth
const RE: f64 = 6_378_137.0; // m, WGS-84 equatorial radius
const J2_COEFF: f64 = 1.082_626_68e-3; // unnormalised J₂

/// Augmented dimension for a 6-state STT system: 6 + 36 + 126 = 168.
const M: usize = augmented_len(6);

/// A bound, out-of-plane LEO with every component nonzero, so the tensors are rich.
fn nominal_state() -> [f64; 6] {
    [6.9e6, 1.0e6, 0.8e6, -1.5e3, 6.8e3, 2.5e3]
}

/// Two-body + J₂ as a `ScalarSystem<6>`, the force differentiated to second order.
fn force() -> Newtonian<(PointMass, J2)> {
    Newtonian((
        PointMass { mu: MU },
        J2 {
            mu: MU,
            re: RE,
            j2: J2_COEFF,
        },
    ))
}

/// Propagate the state, the STM, and the STT to `t1`.
fn propagate_stt(y0: &[f64; 6], t1: f64) -> ([f64; 6], [[f64; 6]; 6], [[[f64; 6]; 6]; 6]) {
    let f = force();
    let sys = VariationalStt::<6, _>::new(&f);
    let aug0: [f64; M] = augment_stt::<6, M>(y0);
    let traj = integrate(
        &Dop853,
        &sys,
        0.0,
        &aug0,
        t1,
        10.0,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-6),
    )
    .expect("integrated the STT-augmented system");
    split_stt::<6, M>(&traj.final_state())
}

/// Propagate the state and the first-order STM to `t1` (the 42-dim variational system).
fn propagate_stm(y0: &[f64; 6], t1: f64) -> [[f64; 6]; 6] {
    let f = force();
    let var = Variational::<6, _>::new(&f);
    let aug0: [f64; 42] = augment::<6, 42>(y0);
    let traj = integrate(
        &Dop853,
        &var,
        0.0,
        &aug0,
        t1,
        10.0,
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-6),
    )
    .expect("integrated the STM-augmented system");
    let (_, phi) = split::<6, 42>(&traj.final_state());
    phi
}

/// Propagate the plain 6-state nonlinearly to `t1`.
fn propagate_state(y0: &[f64; 6], t1: f64) -> [f64; 6] {
    let f = Plain(force());
    integrate(
        &Dop853,
        &f,
        0.0,
        y0,
        t1,
        10.0,
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-6),
    )
    .expect("integrated the plain state")
    .final_state()
}

fn matvec(a: &[[f64; 6]; 6], x: &[f64; 6]) -> [f64; 6] {
    let mut out = [0.0; 6];
    for i in 0..6 {
        for j in 0..6 {
            out[i] += a[i][j] * x[j];
        }
    }
    out
}

fn norm(v: &[f64; 6]) -> f64 {
    v.iter().map(|c| c * c).sum::<f64>().sqrt()
}

#[test]
fn the_stt_matches_a_finite_difference_of_the_stm() {
    let y0 = nominal_state();
    let t1 = 1800.0; // ~⅓ orbit
    let (_, _, psi) = propagate_stt(&y0, t1);

    // Per-axis step, scaled to the position / velocity magnitude of the state.
    let sr = norm(&[y0[0], y0[1], y0[2], 0.0, 0.0, 0.0]);
    let sv = norm(&[0.0, 0.0, 0.0, y0[3], y0[4], y0[5]]);
    let scale = |k: usize| if k < 3 { sr } else { sv };
    let rel = 1e-4;

    // Ψ_fd[i][j][k] = ∂Φ[i][j]/∂x₀[k] by central difference of the STM.
    let mut psi_fd = [[[0.0; 6]; 6]; 6];
    for k in 0..6 {
        let h = rel * scale(k);
        let (mut yp, mut ym) = (y0, y0);
        yp[k] += h;
        ym[k] -= h;
        let phip = propagate_stm(&yp, t1);
        let phim = propagate_stm(&ym, t1);
        for i in 0..6 {
            for j in 0..6 {
                psi_fd[i][j][k] = (phip[i][j] - phim[i][j]) / (2.0 * h);
            }
        }
    }

    // Row-normalised max relative error: each output row i is scaled by its own largest |Ψ|, so the
    // physically negligible near-zero entries cannot dominate the metric.
    let mut worst = 0.0_f64;
    let mut max_psi = 0.0_f64;
    for i in 0..6 {
        let mut row = 0.0_f64;
        for j in 0..6 {
            for k in 0..6 {
                row = row.max(psi[i][j][k].abs());
            }
        }
        max_psi = nan_max(max_psi, row);
        if row == 0.0 {
            continue;
        }
        for j in 0..6 {
            for k in 0..6 {
                worst = nan_max(worst, (psi[i][j][k] - psi_fd[i][j][k]).abs() / row);
                // Ψ is symmetric in (j,k) by construction of the packed storage.
                assert!(psi[i][j][k] == psi[i][k][j], "Ψ must be symmetric in (j,k)");
            }
        }
    }

    println!(
        "STT vs FD-of-STM: max row-normalised relative error {worst:.3e}; max|Ψ| {max_psi:.3e}"
    );
    assert!(
        max_psi > 1e-12,
        "Ψ must be nonzero for a nonlinear (J₂) force"
    );
    assert!(
        worst < 5e-6,
        "STT disagrees with the finite difference of the STM by {worst:.3e} (above the FD floor)"
    );
}

#[test]
fn the_second_order_deviation_map_beats_the_stm() {
    let y0 = nominal_state();
    let t1 = 1800.0;
    let (_, phi, psi) = propagate_stt(&y0, t1);
    let x_nom = propagate_state(&y0, t1);

    // Residual of the STM-only vs the STM+STT prediction for a finite deviation.
    let residuals = |dx0: &[f64; 6]| -> (f64, f64) {
        let yp = {
            let mut y = y0;
            for i in 0..6 {
                y[i] += dx0[i];
            }
            y
        };
        let x_true = propagate_state(&yp, t1);
        let dx_true = {
            let mut d = [0.0; 6];
            for i in 0..6 {
                d[i] = x_true[i] - x_nom[i];
            }
            d
        };
        let dx1 = matvec(&phi, dx0); // STM only
        let dx2 = map_deviation(&phi, &psi, dx0); // STM + STT
        let (mut e1, mut e2) = ([0.0; 6], [0.0; 6]);
        for i in 0..6 {
            e1[i] = dx1[i] - dx_true[i];
            e2[i] = dx2[i] - dx_true[i];
        }
        (norm(&e1), norm(&e2))
    };

    // ~4 km / ~2.7 m/s deviation, and its half.
    let dx_full = [3000.0, -2000.0, 1500.0, 2.0, -1.5, 1.0];
    let dx_half = {
        let mut d = dx_full;
        for c in &mut d {
            *c *= 0.5;
        }
        d
    };
    let (e1_full, e2_full) = residuals(&dx_full);
    let (e1_half, e2_half) = residuals(&dx_half);

    let ratio_full = e2_full / e1_full;
    let ratio_half = e2_half / e1_half;
    println!(
        "deviation map (‖δ‖≈4km): STM residual {e1_full:.3e} m, STT residual {e2_full:.3e} m, ratio {ratio_full:.3e}"
    );
    println!(
        "deviation map (‖δ‖≈2km): STM residual {e1_half:.3e} m, STT residual {e2_half:.3e} m, ratio {ratio_half:.3e}"
    );

    assert!(
        e2_full < 0.1 * e1_full,
        "STT must cut the STM residual by >10× (got {:.2}×)",
        e1_full / e2_full
    );
    // Residual is O(‖δ‖²) for the STM, O(‖δ‖³) for the STT: halving δ should roughly halve the ratio.
    assert!(
        ratio_half < 0.75 * ratio_full,
        "halving δx₀ should shrink the STT/STM residual ratio ({ratio_half:.3e} vs {ratio_full:.3e})"
    );
}

#[test]
fn the_second_order_covariance_matches_gauss_hermite() {
    let y0 = nominal_state();
    let t1 = 1800.0;
    let (_, phi, psi) = propagate_stt(&y0, t1);

    // Diagonal initial covariance: 500 m per position axis, 0.5 m/s per velocity axis.
    let sigma = [500.0, 500.0, 500.0, 0.5, 0.5, 0.5];
    let mut p0 = [[0.0; 6]; 6];
    for i in 0..6 {
        p0[i][i] = sigma[i] * sigma[i];
    }
    let (mean, cov) = map_covariance(&phi, &psi, &p0);

    // Gauss–Hermite (probabilists', 3-point: nodes ±√3, 0; weights 1/6, 2/3, 1/6), exact for the
    // degree-≤4 moments of the quadratic map — an independent oracle for the Park–Scheeres formulas.
    let nodes = [-(3.0_f64.sqrt()), 0.0, 3.0_f64.sqrt()];
    let wts = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
    let mut mean_gh = [0.0; 6];
    let mut second = [[0.0; 6]; 6];
    for g in 0..3usize.pow(6) {
        let mut idx = g;
        let mut w = 1.0;
        let mut x = [0.0; 6];
        for dim in 0..6 {
            let t = idx % 3;
            idx /= 3;
            w *= wts[t];
            x[dim] = sigma[dim] * nodes[t];
        }
        let y = map_deviation(&phi, &psi, &x); // the same quadratic map the moments describe
        for i in 0..6 {
            mean_gh[i] += w * y[i];
            for j in 0..6 {
                second[i][j] += w * y[i] * y[j];
            }
        }
    }

    let mut cov_scale = 0.0_f64;
    for i in 0..6 {
        cov_scale = nan_max(cov_scale, cov[i][i].abs());
    }
    let mut cov_err = 0.0_f64;
    for i in 0..6 {
        for j in 0..6 {
            let cov_gh = second[i][j] - mean_gh[i] * mean_gh[j];
            cov_err = nan_max(cov_err, (cov[i][j] - cov_gh).abs());
        }
    }
    let mut mean_err = 0.0_f64;
    let mut mean_scale = 0.0_f64;
    for i in 0..6 {
        mean_err = nan_max(mean_err, (mean[i] - mean_gh[i]).abs());
        mean_scale = nan_max(mean_scale, mean[i].abs());
    }

    println!(
        "2nd-order covariance vs Gauss–Hermite: cov rel err {:.3e}, mean rel err {:.3e}, mean bias {:.3e} m",
        cov_err / cov_scale,
        mean_err / mean_scale,
        mean_scale
    );
    assert!(
        mean_scale > 0.0,
        "the curvature must bias the mean off zero"
    );
    assert!(
        cov_err / cov_scale < 1e-8,
        "2nd-order covariance disagrees with Gauss–Hermite by {:.3e}",
        cov_err / cov_scale
    );
    assert!(
        mean_err / mean_scale < 1e-7,
        "2nd-order mean shift disagrees with Gauss–Hermite by {:.3e}",
        mean_err / mean_scale
    );
}

/// A purely linear force `a = -k r`: `∂²f/∂x² = 0`, so the STT coupling term vanishes and `Ψ` must
/// stay exactly 0 while the STM stays non-trivial — isolating the `H:ΦΦ` term of the ODE.
struct LinearSpring {
    k: f64,
}
impl ScalarSystem<6> for LinearSpring {
    fn eval<S: Scalar>(&self, _t: S, y: &[S; 6]) -> [S; 6] {
        let k = S::from_f64(self.k);
        [y[3], y[4], y[5], -(k * y[0]), -(k * y[1]), -(k * y[2])]
    }
}

#[test]
fn a_linear_force_has_zero_stt() {
    let spring = LinearSpring { k: 1.3 };
    let sys = VariationalStt::<6, _>::new(&spring);
    let y0 = [1.0, 0.5, -0.3, 0.2, -0.1, 0.4];
    let aug0: [f64; M] = augment_stt::<6, M>(&y0);
    let traj = integrate(
        &Dop853,
        &sys,
        0.0,
        &aug0,
        2.0,
        0.01,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-9),
    )
    .expect("integrated the linear STT system");
    let (_, phi, psi) = split_stt::<6, M>(&traj.final_state());

    let mut max_psi = 0.0_f64;
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                max_psi = nan_max(max_psi, psi[i][j][k].abs());
            }
        }
    }
    // Off-diagonal STM coupling must be non-trivial (the flow is not the identity).
    let mut off = 0.0_f64;
    for i in 0..6 {
        for j in 0..6 {
            if i != j {
                off = nan_max(off, phi[i][j].abs());
            }
        }
    }
    println!("linear force: max|Ψ| {max_psi:.3e}, max off-diagonal |Φ| {off:.3e}");
    assert!(off > 0.1, "the STM must be non-trivial");
    assert!(
        max_psi < 1e-9,
        "a linear force must give Ψ = 0, got {max_psi:.3e}"
    );
}
