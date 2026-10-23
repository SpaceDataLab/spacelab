//! Adams–Moulton (order 5) in Nordsieck form — a multistep method whose step changes for free.
//!
//! # The Nordsieck idea
//!
//! A classical multistep method stores past *function values* on an equally-spaced grid, so changing
//! the step means re-interpolating the whole history and changing the order means switching to a
//! different coefficient set. The **Nordsieck** representation stores instead the scaled derivatives
//! at a single point — `z = [y, h y', (h²/2) y'', …, (hᵠ/q!) y⁽ᵠ⁾]`. In this form the two operations
//! a production solver needs constantly become trivial:
//!
//! * **change the step** `h → h'` by rescaling `zᵢ ← (h'/h)ⁱ zᵢ` (no re-interpolation);
//! * **change the order** by lengthening or shortening `z` and swapping the correction vector.
//!
//! A step is predict–evaluate–correct: the predictor is the Pascal (Taylor-shift) matrix
//! `z_pred = P z` (exact for a degree-`q` polynomial), then one force evaluation gives the correction
//! `e = h f(t+h, y_pred) − z_pred[1]`, and `z ← z_pred + ℓ e` with the order-5 Nordsieck l-vector.
//! That `ℓ` is *derived* in `derivations/adams_nordsieck.mac`, not tabulated: it is pinned by the
//! requirement that the one-step matrix `M = P − ℓ (row₁ P)` have characteristic polynomial
//! `x⁵(x−1)` — principal root 1, all spurious roots 0, so the method is stable — with `ℓ₀` fixed by
//! consistency to the Adams–Moulton coefficient `β₀ = 251/720`. (A l-vector with a spurious root off
//! the origin blows up; that is exactly what a mis-transcribed value did before it was derived.)
//!
//! This integrator runs at order 5 with the step chosen by the caller; the state lives in a
//! [`Nordsieck`] struct whose [`rescale`](Nordsieck::rescale) makes a step change a single `O(q)`
//! multiply (the operation an LSODE-style variable-step driver leans on), and whose scaled top
//! component `(hᵠ/q!) y⁽ᵠ⁾` is the natural local-error estimate. The startup vector comes from the
//! exact Taylor jets of [`crate::taylor`], so the method runs at full order from the first step.

// Full-precision l-vector literals; the loops index Nordsieck components.
#![allow(
    clippy::needless_range_loop,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

use crate::driver::{IntegrationError, step_count};
use crate::taylor::solution_jets;
use crate::variational::ScalarSystem;

const MAXQ: usize = 5;
const SLOTS: usize = MAXQ + 1; // Nordsieck vector length for order q = MAXQ

/// The order-5 Nordsieck-Adams l-vector `[ℓ₀…ℓ₅]`, derived and verified in
/// `derivations/adams_nordsieck.mac`: `ℓ₀ = 251/720` (the Adams–Moulton `β₀`), `ℓ₁ = 1`,
/// `ℓ₅ = 1/5!`, and `ℓ₂…ℓ₄` are the unique values making the method's spurious roots all zero.
const L5: [f64; MAXQ + 1] = [
    251.0 / 720.0,
    1.0,
    25.0 / 24.0,
    35.0 / 72.0,
    5.0 / 48.0,
    1.0 / 120.0,
];

/// Binomial coefficients `C(j, i)` for the Pascal predictor.
const fn binom() -> [[f64; SLOTS]; SLOTS] {
    let mut c = [[0.0; SLOTS]; SLOTS];
    let mut j = 0;
    while j < SLOTS {
        c[j][0] = 1.0;
        let mut i = 1;
        while i <= j {
            c[j][i] = c[j - 1][i - 1] + c[j - 1][i];
            i += 1;
        }
        j += 1;
    }
    c
}
const BINOM: [[f64; SLOTS]; SLOTS] = binom();

/// The order-5 Nordsieck state — the scaled derivatives `[y, h y', …, (h⁵/5!) y⁽⁵⁾]` per component,
/// with the step `h` they are scaled to. Advancing it is one predict–evaluate–correct; changing the
/// step is a rescale, so it is the object a variable-step driver carries between steps.
#[derive(Debug, Clone, Copy)]
pub struct Nordsieck<const N: usize> {
    z: [[f64; MAXQ + 1]; N],
    h: f64,
}

impl<const N: usize> Nordsieck<N> {
    /// Build the order-5 Nordsieck state at `(t, y)` for step `h`, seeding the scaled derivatives from
    /// the exact Taylor jets (`zᵢ = hⁱ · jetᵢ`).
    #[must_use]
    pub fn start(force: &impl ScalarSystem<N>, t: f64, y: &[f64; N], h: f64) -> Self {
        let jets = solution_jets::<N, { MAXQ + 1 }>(force, t, y);
        let mut z = [[0.0; MAXQ + 1]; N];
        for c in 0..N {
            let mut hp = 1.0;
            for i in 0..=MAXQ {
                z[c][i] = hp * jets[c][i];
                hp *= h;
            }
        }
        Self { z, h }
    }

    /// The current state value `y`.
    #[must_use]
    pub fn value(&self) -> [f64; N] {
        let mut y = [0.0; N];
        for c in 0..N {
            y[c] = self.z[c][0];
        }
        y
    }

    /// Rescale to a new step `h_new`: `zᵢ ← (h_new/h)ⁱ zᵢ`. This is what makes the step *variable* at
    /// no cost — no history re-interpolation, unlike an ordinate multistep method.
    pub fn rescale(&mut self, h_new: f64) {
        let ratio = h_new / self.h;
        for c in 0..N {
            let mut r = 1.0;
            for i in 0..=MAXQ {
                self.z[c][i] *= r;
                r *= ratio;
            }
        }
        self.h = h_new;
    }

    /// One predict–evaluate–correct step of the current step `h` from time `t`. Returns the RMS
    /// correction (a local-error indicator).
    pub fn step(&mut self, force: &impl ScalarSystem<N>, t: f64) -> f64 {
        // Predict: z_pred[c][i] = Σ_{j≥i} C(j,i) z[c][j].
        let mut zp = [[0.0; MAXQ + 1]; N];
        for c in 0..N {
            for i in 0..=MAXQ {
                let mut acc = 0.0;
                for j in i..=MAXQ {
                    acc += BINOM[j][i] * self.z[c][j];
                }
                zp[c][i] = acc;
            }
        }
        // Evaluate and correct with the order-5 l-vector.
        let mut yp = [0.0; N];
        for c in 0..N {
            yp[c] = zp[c][0];
        }
        let f = force.eval::<f64>(t + self.h, &yp);
        let mut e = [0.0; N];
        for c in 0..N {
            e[c] = self.h * f[c] - zp[c][1];
        }
        for c in 0..N {
            for i in 0..=MAXQ {
                self.z[c][i] = zp[c][i] + L5[i] * e[c];
            }
        }
        let mut acc = 0.0;
        for c in 0..N {
            acc += (e[c] / (1.0 + yp[c].abs())).powi(2);
        }
        (acc / N as f64).sqrt()
    }
}

/// Integrate `y' = force(t, y)` from `t0` to `t1` with the order-5 Adams–Nordsieck method at step `h`
/// (adjusted so an integer number of steps spans the interval). For a variable step, drive the
/// [`Nordsieck`] state directly and call [`Nordsieck::rescale`] between steps.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] if the span or step is not positive.
pub fn integrate_adams_nordsieck<const N: usize>(
    force: &impl ScalarSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    h: f64,
) -> Result<[f64; N], IntegrationError> {
    if !(t1 > t0 && h > 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    let n_steps = step_count(t0, t1, h, 1)?;
    let step = (t1 - t0) / n_steps as f64;

    let mut ns = Nordsieck::start(force, t0, y0, step);
    let mut t = t0;
    for _ in 0..n_steps {
        ns.step(force, t);
        t += step;
    }
    Ok(ns.value())
}
