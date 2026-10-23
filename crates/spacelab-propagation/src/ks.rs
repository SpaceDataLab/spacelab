//! Kustaanheimo–Stiefel regularization: integrate the Kepler problem as a linear oscillator, with no
//! singularity at `r → 0` and uniform steps in fictitious time.
//!
//! # Why regularize
//!
//! Near perihelion of a high-eccentricity orbit — or during a close encounter — the acceleration
//! `μ/r²` spikes and the velocity turns fast, so a fixed-step Cartesian integrator either loses all
//! its accuracy there or must crawl. The KS transformation cures this at the source: it lifts the 3-D
//! position to a 4-D spinor `u` (`r = |u|²`, so `r = 0` is a smooth point, not a pole) and rescales
//! time by `dt = r ds` (the Sundman transform). In the new variables the *unperturbed* two-body
//! equation of motion is exactly
//!
//! ```text
//! u'' = (h/2) u,     h = ½|v|² − μ/r  (the Kepler energy, < 0 for a bound orbit),
//! ```
//!
//! a **linear harmonic oscillator** — no singularity, and uniform steps in `s` spread evenly around
//! the orbit in eccentric anomaly instead of bunching at perihelion. A perturbation `P` adds the
//! forcing `(r/2) L(u)ᵀ P` and lets the energy `h` drift by `2 u'ᵀ L(u)ᵀ P`.
//!
//! The KS matrix `L(u)` and the identities used here — `|r| = |u|²`, `L(u)ᵀL(u) = |u|² I`, and the
//! bilinear relation — are checked in `derivations/ks_regularization.mac`.

// The state is indexed by component; the step count casts a small integer.
#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

use crate::driver::IntegrationError;
use crate::forces::Acceleration;

const N: usize = 10; // state = [u(4), u'(4), t, h]

/// `L(u) · w` — the KS matrix applied to a 4-vector.
fn lu(u: &[f64; 4], w: &[f64; 4]) -> [f64; 4] {
    [
        u[0] * w[0] - u[1] * w[1] - u[2] * w[2] + u[3] * w[3],
        u[1] * w[0] + u[0] * w[1] - u[3] * w[2] - u[2] * w[3],
        u[2] * w[0] + u[3] * w[1] + u[0] * w[2] + u[1] * w[3],
        u[3] * w[0] - u[2] * w[1] + u[1] * w[2] - u[0] * w[3],
    ]
}

/// `L(u)ᵀ · w`.
fn ltu(u: &[f64; 4], w: &[f64; 4]) -> [f64; 4] {
    [
        u[0] * w[0] + u[1] * w[1] + u[2] * w[2] + u[3] * w[3],
        -u[1] * w[0] + u[0] * w[1] + u[3] * w[2] - u[2] * w[3],
        -u[2] * w[0] - u[3] * w[1] + u[0] * w[2] + u[1] * w[3],
        u[3] * w[0] - u[2] * w[1] + u[1] * w[2] - u[0] * w[3],
    ]
}

/// KS position/velocity from the spinor: `r = (first 3 of L(u)u)`, `v = (2/|u|²)(first 3 of L(u)u')`.
fn to_cartesian(u: &[f64; 4], up: &[f64; 4]) -> ([f64; 3], [f64; 3]) {
    let ruu = lu(u, u);
    let r = u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + u[3] * u[3];
    let luv = lu(u, up);
    (
        [ruu[0], ruu[1], ruu[2]],
        [2.0 * luv[0] / r, 2.0 * luv[1] / r, 2.0 * luv[2] / r],
    )
}

/// Spinor `(u, u')` and Kepler energy `h` from Cartesian state (the standard branch-selected inverse).
fn from_cartesian(r: &[f64; 3], v: &[f64; 3], mu: f64) -> ([f64; 4], [f64; 4], f64) {
    let rmag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    let u = if r[0] >= 0.0 {
        let u0 = f64::midpoint(rmag, r[0]).sqrt();
        [u0, r[1] / (2.0 * u0), r[2] / (2.0 * u0), 0.0]
    } else {
        let u1 = ((rmag - r[0]) / 2.0).sqrt();
        [r[1] / (2.0 * u1), u1, 0.0, r[2] / (2.0 * u1)]
    };
    let v4 = [v[0], v[1], v[2], 0.0];
    let ltv = ltu(&u, &v4);
    let up = [0.5 * ltv[0], 0.5 * ltv[1], 0.5 * ltv[2], 0.5 * ltv[3]];
    let h = 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - mu / rmag;
    (u, up, h)
}

/// The regularized right-hand side `d/ds` of `[u, u', t, h]`.
fn rhs<A: Acceleration>(pert: &A, s: &[f64; N]) -> [f64; N] {
    let u = [s[0], s[1], s[2], s[3]];
    let up = [s[4], s[5], s[6], s[7]];
    let (t, h) = (s[8], s[9]);
    let r = u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + u[3] * u[3];
    let (rvec, vvec) = to_cartesian(&u, &up);
    let p = pert.acceleration::<f64>(t, &rvec, &vvec);
    let p4 = [p[0], p[1], p[2], 0.0];
    let ltp = ltu(&u, &p4);
    let mut upp = [0.0; 4];
    for i in 0..4 {
        upp[i] = 0.5 * h * u[i] + 0.5 * r * ltp[i];
    }
    let hp = 2.0 * (up[0] * ltp[0] + up[1] * ltp[1] + up[2] * ltp[2] + up[3] * ltp[3]);
    [
        up[0], up[1], up[2], up[3], upp[0], upp[1], upp[2], upp[3], r, hp,
    ]
}

fn rk4_step<A: Acceleration>(pert: &A, s: &[f64; N], ds: f64) -> [f64; N] {
    let add = |a: &[f64; N], f: f64, b: &[f64; N]| {
        let mut o = *a;
        for i in 0..N {
            o[i] += f * b[i];
        }
        o
    };
    let k1 = rhs(pert, s);
    let k2 = rhs(pert, &add(s, 0.5 * ds, &k1));
    let k3 = rhs(pert, &add(s, 0.5 * ds, &k2));
    let k4 = rhs(pert, &add(s, ds, &k3));
    let mut out = *s;
    for i in 0..N {
        out[i] += ds / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    out
}

/// Integrate the two-body problem about `mu` under perturbation `pert`, in KS variables over a span
/// `s_span` of fictitious time with `n_steps` uniform steps. Returns the final `(r, v)` and the
/// elapsed real time.
///
/// For a bound orbit one full revolution is `s_span = π/√(−h/2)` — half the spinor oscillator's
/// period, since `u` and `−u` map to the same position. `h` is the Kepler energy of the initial state.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] if `s_span`, `n_steps`, or `mu` is not positive, or the initial
/// energy is non-negative (KS regularization here is for bound orbits).
pub fn integrate_ks<A: Acceleration>(
    mu: f64,
    pert: &A,
    r0: &[f64; 3],
    v0: &[f64; 3],
    s_span: f64,
    n_steps: usize,
) -> Result<([f64; 3], [f64; 3], f64), IntegrationError> {
    if !(s_span > 0.0 && mu > 0.0) || n_steps == 0 {
        return Err(IntegrationError::BadArguments);
    }
    let (u, up, h) = from_cartesian(r0, v0, mu);
    if h >= 0.0 {
        return Err(IntegrationError::BadArguments);
    }
    let mut s = [u[0], u[1], u[2], u[3], up[0], up[1], up[2], up[3], 0.0, h];
    let ds = s_span / n_steps as f64;
    for _ in 0..n_steps {
        s = rk4_step(pert, &s, ds);
    }
    let (rvec, vvec) = to_cartesian(&[s[0], s[1], s[2], s[3]], &[s[4], s[5], s[6], s[7]]);
    Ok((rvec, vvec, s[8]))
}

#[cfg(test)]
mod tests {
    use super::{ltu, lu};

    /// A spinor with four distinct non-zero components of unlike sign. The identities below are
    /// scale-free, so `O(1)` keeps the residual readable as a relative error.
    const U: [f64; 4] = [1.3, -0.7, 2.1, 0.45];

    /// The transcription of `L(u)` into Rust is pinned by the identities the derivation proves, not
    /// by any propagation.
    ///
    /// `derivations/ks_regularization.mac` proves `L(u)ᵀL(u) = |u|²I₄` and that the fourth component
    /// of `L(u)u` vanishes — but it proves them of the *symbolic* matrix. Nothing checked that the
    /// sixteen entries here match that matrix, and the gap is not hypothetical: `to_cartesian` reads
    /// only the first three components of both products it takes, so the whole fourth row of `lu` is
    /// computed and discarded, and every caller of `ltu` passes a 4-vector whose last component is a
    /// structural zero. Eighteen mutants of those two functions survived the whole integration suite
    /// for that reason. Under `L(u)ᵀL(u) = |u|²I` every entry carries, because a wrong `L₃ₖ` shows up
    /// in the whole `k`-th column against the untouched `L₃ⱼ`.
    #[test]
    fn the_ks_matrix_transcription_satisfies_the_identities_the_derivation_proves() {
        let usq = U[0] * U[0] + U[1] * U[1] + U[2] * U[2] + U[3] * U[3];

        // The four basis vectors build `L(u)ᵀL(u)` a column at a time; the generic one is what
        // separates `a * b` from `a / b`, which agree on a column of ones and zeros.
        let probes = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.31, -1.7, 0.86, -2.3],
        ];
        for w in probes {
            let got = ltu(&U, &lu(&U, &w));
            for k in 0..4 {
                let want = usq * w[k];
                let err = (got[k] - want).abs();
                assert!(
                    err < 1e-14 * usq * (1.0 + w[k].abs()),
                    "LᵀL is not |u|²I at column {w:?}, row {k}: {:e} vs {want:e}",
                    got[k]
                );
            }
        }

        // Identity 2: the fourth component of `L(u)u` is the KS constraint and vanishes identically.
        let constraint = lu(&U, &U)[3];
        assert!(
            constraint.abs() < 1e-14 * usq,
            "the KS constraint does not vanish: {constraint:e}"
        );
    }
}
