//! MRI-GARK-ERK33a — a multirate infinitesimal integrator of order 3 (Sandu 2019).
//!
//! # Different clocks for different physics
//!
//! A satellite simulation runs several processes at once whose natural time steps differ by orders of
//! magnitude: the orbit evolves over minutes, an attitude loop over milliseconds, a controller faster
//! still. Stepping everything at the fastest rate wastes enormous effort on the slow part. A
//! **multirate** method steps the slow part `f_S` on a large macro-step `H` and the fast part `f_F`
//! on many small sub-steps *inside* each slow stage — each subsystem on its own clock, coupled to
//! third order.
//!
//! # How it couples
//!
//! One macro-step advances through three slow stages. At each stage the fast ODE is integrated over a
//! sub-interval with a *modified* right-hand side: the true fast term `f_F` plus a time-dependent
//! forcing `Σⱼ γ_{i,j}(τ) f_S(Yⱼ)` built from the slow derivatives at the stages computed so far. The
//! coupling weights `γ_{i,j}(τ) = Γ⁰_{i,j} + Γ¹_{i,j} τ` (polynomials in the normalised fast time `τ`)
//! are the exact rationals of the reference method (Sandu 2019 / SUNDIALS `ARKODE_MRI_GARK_ERK33a`);
//! `derivations/mri_gark_order.mac` checks internal consistency (`Γ⁰·1 = Δc`, `Γ¹·1 = 0`) and the
//! order-3 coupling condition. The inner fast solve here is the classical RK4 with `m` sub-steps.

#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

use crate::driver::{IntegrationError, step_count};

const DELTA_C: f64 = 1.0 / 3.0; // each slow stage spans Δc·H of real time
const C_NODES: [f64; 4] = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];

/// Constant coupling coefficients `Γ⁰_{i,j}` (stage row `i`, slow-stage column `j`), 0-based.
const G0: [[f64; 3]; 3] = [
    [1.0 / 3.0, 0.0, 0.0],
    [-1.0 / 3.0, 2.0 / 3.0, 0.0],
    [0.0, -2.0 / 3.0, 1.0],
];
/// Linear-in-`τ` coupling coefficients `Γ¹_{i,j}`.
const G1: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, -0.5]];

fn axpy<const N: usize>(y: &[f64; N], a: f64, d: &[f64; N]) -> [f64; N] {
    let mut o = *y;
    for k in 0..N {
        o[k] += a * d[k];
    }
    o
}

/// Integrate `y' = fast(t,y) + slow(t,y)` from `t0` to `t1` with MRI-GARK-ERK33a: the slow part on
/// macro-steps of size `big_h`, the fast part on `m` inner RK4 sub-steps per slow stage. `fast` and
/// `slow` return the respective parts of the derivative.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] if the span, macro-step, or sub-step count is not positive.
pub fn integrate_mri<const N: usize, Ff, Fs>(
    fast: &Ff,
    slow: &Fs,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    big_h: f64,
    m: usize,
) -> Result<[f64; N], IntegrationError>
where
    Ff: Fn(f64, &[f64; N]) -> [f64; N],
    Fs: Fn(f64, &[f64; N]) -> [f64; N],
{
    if !(t1 > t0 && big_h > 0.0) || m == 0 {
        return Err(IntegrationError::BadArguments);
    }
    let n_steps = step_count(t0, t1, big_h, 1)?;
    let h = (t1 - t0) / n_steps as f64;

    let mut y = *y0;
    let mut t = t0;
    for _ in 0..n_steps {
        // Slow-stage values Y[0..3] (Y[0] = y) and the slow derivatives f_S(Y_j) computed as we go.
        let mut yj = [y; 3];
        let mut fsj = [[0.0; N]; 3];
        fsj[0] = slow(t, &y);

        let mut y_end = y;
        for s in 0..3 {
            // Inner fast solve over θ ∈ [0, H]: dv/dθ = Δc·f_F(t_real, v) + Σⱼ γ_{s,j}(θ/H) f_S(Yⱼ).
            let base_t = t + C_NODES[s] * h;
            let dtheta = h / m as f64;
            let mut v = yj[s];
            let g = |theta: f64, v: &[f64; N]| -> [f64; N] {
                let tau = theta / h;
                let ff = fast(base_t + DELTA_C * theta, v);
                let mut out = [0.0; N];
                for k in 0..N {
                    let mut force = 0.0;
                    for j in 0..=s {
                        force += (G0[s][j] + G1[s][j] * tau) * fsj[j][k];
                    }
                    out[k] = DELTA_C * ff[k] + force;
                }
                out
            };
            for step in 0..m {
                let th = step as f64 * dtheta;
                let k1 = g(th, &v);
                let k2 = g(th + 0.5 * dtheta, &axpy(&v, 0.5 * dtheta, &k1));
                let k3 = g(th + 0.5 * dtheta, &axpy(&v, 0.5 * dtheta, &k2));
                let k4 = g(th + dtheta, &axpy(&v, dtheta, &k3));
                for k in 0..N {
                    v[k] += dtheta / 6.0 * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
                }
            }
            y_end = v;
            if s < 2 {
                yj[s + 1] = v;
                fsj[s + 1] = slow(t + C_NODES[s + 1] * h, &v);
            }
        }
        y = y_end;
        t += h;
    }
    Ok(y)
}
