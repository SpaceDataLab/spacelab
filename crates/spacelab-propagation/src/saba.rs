//! SABA_n — high-order symplectic integrators for a near-integrable Hamiltonian `H = A + εB`
//! (Laskar & Robutel 2001), with `A` the exact Kepler drift and `B` the perturbation kick.
//!
//! # Beyond WHFast
//!
//! [WHFast](crate::whfast) splits a perturbed orbit into an exact Kepler drift and a perturbation
//! kick and composes them as second-order leapfrog. SABA_n keeps that splitting but composes the
//! drifts and kicks with the **Gauss–Legendre** coefficients of order `n`: the method is order `2n`
//! in the step *and* stays second order in the perturbation strength `ε`. For a nearly-Keplerian
//! orbit — where `ε` is tiny (`J₂ ~ 10⁻³`) — this delivers far more accuracy per force evaluation
//! than plain WHFast, while remaining symplectic (bounded energy error). SABA₁ is exactly leapfrog;
//! SABA₂ is order 4, SABA₄ order 8.
//!
//! The structure of one step is a palindrome of `n+1` drifts and `n` kicks,
//! `drift(c₀h) kick(d₀h) drift(c₁h) … kick(d_{n-1}h) drift(cₙh)`, with the drift lengths `cᵢ` and
//! kick weights `dᵢ` derived from the Gauss–Legendre quadrature in `derivations/saba_coeffs.mac`
//! (both sum to 1; SABA₂'s `c = (½ ∓ 1/(2√3), 1/√3)` closed form checked). The drift is the exact
//! universal-variable Kepler advance shared with WHFast; the kick applies the perturbing acceleration
//! as a velocity impulse.

// The Gauss–Legendre coefficients are full-precision literals; the step count casts a small integer.
#![allow(
    clippy::needless_range_loop,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    clippy::doc_markdown
)]

use crate::driver::{IntegrationError, step_count};
use crate::forces::Acceleration;
use crate::whfast::kepler_drift;

/// SABA₂ drift lengths `c` (order 4): `½ ∓ 1/(2√3)`, `1/√3`.
const SABA2_C: [f64; 3] = [
    0.211324865405187117745425609749021272,
    0.577350269189625764509148780501957456,
    0.211324865405187117745425609749021272,
];
/// SABA₂ kick weights `d`: the Gauss–Legendre-2 weights.
const SABA2_D: [f64; 2] = [0.5, 0.5];

/// SABA₄ drift lengths `c` (order 8), from the Gauss–Legendre-4 abscissae.
const SABA4_C: [f64; 5] = [
    0.0694318442029737123880267555535952475,
    0.260577634004598155210640364894782409,
    0.339981043584856264802665759103244687,
    0.260577634004598155210640364894782409,
    0.0694318442029737123880267555535952475,
];
/// SABA₄ kick weights `d`: the Gauss–Legendre-4 weights.
const SABA4_D: [f64; 4] = [
    0.173927422568726928686531974610999706,
    0.326072577431273071313468025389000294,
    0.326072577431273071313468025389000294,
    0.173927422568726928686531974610999706,
];

/// One SABA step of size `h`: the palindrome `drift(c₀h) kick(d₀h) … kick(d_{n-1}h) drift(cₙh)`.
fn saba_step<A: Acceleration>(
    mu: f64,
    pert: &A,
    r: &mut [f64; 3],
    v: &mut [f64; 3],
    t: f64,
    h: f64,
    c: &[f64],
    d: &[f64],
) -> Result<(), IntegrationError> {
    let mut tl = t;
    kepler_drift(mu, tl, r, v, c[0] * h)?;
    tl += c[0] * h;
    for i in 0..d.len() {
        let a = pert.acceleration::<f64>(tl, r, v);
        for k in 0..3 {
            v[k] += d[i] * h * a[k];
        }
        kepler_drift(mu, tl, r, v, c[i + 1] * h)?;
        tl += c[i + 1] * h;
    }
    Ok(())
}

fn integrate_saba<A: Acceleration>(
    mu: f64,
    pert: &A,
    r0: &[f64; 3],
    v0: &[f64; 3],
    t0: f64,
    t1: f64,
    dt: f64,
    c: &[f64],
    d: &[f64],
) -> Result<([f64; 3], [f64; 3]), IntegrationError> {
    if !(t1 > t0 && dt > 0.0 && mu > 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    let n_steps = step_count(t0, t1, dt, 1)?;
    let step = (t1 - t0) / n_steps as f64;
    let mut r = *r0;
    let mut v = *v0;
    let mut t = t0;
    for _ in 0..n_steps {
        saba_step(mu, pert, &mut r, &mut v, t, step, c, d)?;
        t += step;
    }
    Ok((r, v))
}

/// Integrate a satellite about central mass `mu` under a small perturbation `pert` with SABA₂
/// (order 4). `pert` is the non-central acceleration; the step spans `[t0, t1]` in integer steps.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] if the span, step, or `mu` is not positive, or if the state
/// stops having an osculating orbit for the Kepler drift to advance.
pub fn integrate_saba2<A: Acceleration>(
    mu: f64,
    pert: &A,
    r0: &[f64; 3],
    v0: &[f64; 3],
    t0: f64,
    t1: f64,
    dt: f64,
) -> Result<([f64; 3], [f64; 3]), IntegrationError> {
    integrate_saba(mu, pert, r0, v0, t0, t1, dt, &SABA2_C, &SABA2_D)
}

/// Integrate with SABA₄ (order 8) — the same splitting as [`integrate_saba2`] with higher-order
/// Gauss–Legendre composition, for much lower error per force evaluation on a near-Keplerian orbit.
///
/// # Errors
///
/// As [`integrate_saba2`].
pub fn integrate_saba4<A: Acceleration>(
    mu: f64,
    pert: &A,
    r0: &[f64; 3],
    v0: &[f64; 3],
    t0: f64,
    t1: f64,
    dt: f64,
) -> Result<([f64; 3], [f64; 3]), IntegrationError> {
    integrate_saba(mu, pert, r0, v0, t0, t1, dt, &SABA4_C, &SABA4_D)
}
