//! Richardson's third-order analytic halo-orbit approximation — a robust initial guess for the
//! differential corrector ([`crate::periodic::halo_orbit`]).
//!
//! Given only the mass ratio, a collinear point (L1 or L2), and the out-of-plane amplitude `A_z`,
//! this constructs the perpendicular-crossing synodic state of the halo without any numerical search.
//! It is the standard seed that turns halo generation from "hunt for a guess" into "correct a good
//! one", and it works for both the northern (`δ_n = +1`) and southern (`δ_n = −1`) families.
//!
//! The construction (Richardson, *Celest. Mech.* 22, 1980; Koon–Lo–Marsden–Ross, appendix) expands
//! the motion about the libration point in the amplitudes to third order. The in-plane frequency `λ`
//! and ratio `k` come from the linearised characteristic equation; the coefficients `a_ij, b_ij, d_ij`
//! and the frequency/amplitude relations `s_i, l_i` are the harmonic-balance solution.
//!
//! Every formula here is proven — not transcribed — in `derivations/richardson_halo.mac` (Maxima, the
//! authoritative CAS for this repo). Maxima independently re-derives `λ` and `k` (characteristic root
//! and first-order harmonic balance), builds the Legendre forcing from the Bonnet recurrence, and
//! substitutes the full third-order series back into the exact equations of motion in the
//! Lindstedt–Poincaré bookkeeping the construction assumes — amplitudes `O(ε)`, detuning
//! `λ² − c₂ = O(ε²)`, which is what the amplitude constraint asserts. Orders `ε` and `ε²` vanish in every
//! harmonic of all three equations, and so does every harmonic of `ε³` except the resonant `cos θ` of the
//! x-equation and `sin θ` of the y-equation — those two cannot vanish termwise, because the linear
//! operator at frequency `λ` is singular by construction and its kernel is the first-order solution
//! itself. What they satisfy instead is the solvability condition `P_x = k P_y`, the projection onto the
//! operator's left null vector, and that is the identity `s_i` and `l_i` are determined by. All of it is
//! measured to ~1e-15 against an `ε⁴` truncation of order 6, for Earth–Moon L1 and L2. The Rust
//! coefficients are checked against Maxima's numeric values to 1e-9, and the assembled seed is shown to
//! converge under the true-CR3BP corrector (`derivations/richardson_halo.py` cross-checks the synodic
//! mapping by integrating the periodicity defect; its numbers agree with Maxima to every digit).
//!
//! Nondimensional CR3BP units (synodic frame, separation and mean motion 1). The amplitude `A_z` is
//! Richardson's γ-normalized out-of-plane amplitude (the natural amplitude of the expansion); the
//! resulting synodic out-of-plane excursion is `z0 ≈ γ·A_z`. Useful `A_z` values run to a few tenths;
//! third-order accuracy degrades as the amplitude grows, but the corrector absorbs the residual.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::manual_midpoint
)]

use crate::cr3bp::Cr3bp;

#[inline]
fn ipow(base: f64, exp: u32) -> f64 {
    let mut acc = 1.0;
    for _ in 0..exp {
        acc *= base;
    }
    acc
}

/// The Richardson `c_n` constant for the collinear point (`li = 0` → L1, `li = 1` → L2).
fn c_n(mu: f64, gamma: f64, li: usize, n: u32) -> f64 {
    let sign = if n.is_multiple_of(2) { 1.0 } else { -1.0 };
    let g3 = gamma * gamma * gamma;
    let gp = ipow(gamma, n + 1);
    if li == 0 {
        (mu + sign * (1.0 - mu) * gp / ipow(1.0 - gamma, n + 1)) / g3
    } else {
        (sign * mu + sign * (1.0 - mu) * gp / ipow(1.0 + gamma, n + 1)) / g3
    }
}

/// The full third-order coefficient set for a collinear point, all functions of `(μ, γ, L_i)`.
/// Crate-internal: the public surface is [`richardson_halo_guess`] and [`crate::periodic::richardson_halo`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Coefficients {
    /// Distance from the libration point to the nearer primary.
    pub gamma: f64,
    /// Richardson constant `c_2` (the linearised-dynamics coefficient).
    pub c2: f64,
    /// In-plane frequency `λ` and amplitude ratio `k`.
    pub lambda: f64,
    pub k: f64,
    /// Second-order amplitude coefficients.
    pub a21: f64,
    pub a22: f64,
    pub a23: f64,
    pub a24: f64,
    pub b21: f64,
    pub b22: f64,
    pub d21: f64,
    /// Third-order amplitude coefficients.
    pub a31: f64,
    pub a32: f64,
    pub b31: f64,
    pub b32: f64,
    pub d31: f64,
    pub d32: f64,
    /// Frequency-correction and amplitude-constraint coefficients.
    pub s1: f64,
    pub s2: f64,
    pub l1: f64,
    pub l2: f64,
}

impl Coefficients {
    /// Build the coefficient set for `sys` at collinear point `li` (0 → L1, 1 → L2). Returns `None`
    /// for any other index.
    pub(crate) fn new(sys: Cr3bp, li: usize) -> Option<Self> {
        if li > 1 {
            return None;
        }
        let mu = sys.mu;
        let x_li = sys.collinear_points()[li].x;
        // γ = distance from L_i to the secondary at x = 1 − μ.
        let gamma = libm::fabs(x_li - (1.0 - mu));

        let c2 = c_n(mu, gamma, li, 2);
        let c3 = c_n(mu, gamma, li, 3);
        let c4 = c_n(mu, gamma, li, 4);

        let lambda = libm::sqrt((2.0 - c2 + libm::sqrt(9.0 * c2 * c2 - 8.0 * c2)) / 2.0);
        let l2f = lambda * lambda;
        let k = (l2f + 1.0 + 2.0 * c2) / (2.0 * lambda);

        let d1 = 3.0 * l2f / k * (k * (6.0 * l2f - 1.0) - 2.0 * lambda);
        let d2 = 8.0 * l2f / k * (k * (11.0 * l2f - 1.0) - 2.0 * lambda);

        let a21 = 3.0 * c3 * (k * k - 2.0) / (4.0 * (1.0 + 2.0 * c2));
        let a22 = 3.0 * c3 / (4.0 * (1.0 + 2.0 * c2));
        let a23 = -3.0 * c3 * lambda / (4.0 * k * d1)
            * (3.0 * ipow(k, 3) * lambda - 6.0 * k * (k - lambda) + 4.0);
        let a24 = -3.0 * c3 * lambda / (4.0 * k * d1) * (2.0 + 3.0 * k * lambda);
        let b21 = -3.0 * c3 * lambda / (2.0 * d1) * (3.0 * k * lambda - 4.0);
        let b22 = 3.0 * c3 * lambda / d1;
        let d21 = -c3 / (2.0 * l2f);

        let a31 = -9.0 * lambda / (4.0 * d2)
            * (4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + k * k))
            + (9.0 * l2f + 1.0 - c2) / (2.0 * d2)
                * (3.0 * c3 * (2.0 * a23 - k * b21) + c4 * (2.0 + 3.0 * k * k));
        let a32 = -9.0 * lambda / (4.0 * d2) * (4.0 * c3 * (k * a24 - b22) + k * c4)
            - 3.0 * (9.0 * l2f + 1.0 - c2) / (2.0 * d2) * (c3 * (k * b22 + d21 - 2.0 * a24) - c4);
        let b31 = 3.0 / (8.0 * d2)
            * 8.0
            * lambda
            * (3.0 * c3 * (k * b21 - 2.0 * a23) - c4 * (2.0 + 3.0 * k * k))
            + 3.0 / (8.0 * d2)
                * (9.0 * l2f + 1.0 + 2.0 * c2)
                * (4.0 * c3 * (k * a23 - b21) + k * c4 * (4.0 + k * k));
        let b32 = 9.0 * lambda / d2 * (c3 * (k * b22 + d21 - 2.0 * a24) - c4)
            + 3.0 * (9.0 * l2f + 1.0 + 2.0 * c2) / (8.0 * d2)
                * (4.0 * c3 * (k * a24 - b22) + k * c4);
        let d31 = 3.0 / (64.0 * l2f) * (4.0 * c3 * a24 + c4);
        let d32 = 3.0 / (64.0 * l2f) * (4.0 * c3 * (a23 - d21) + c4 * (4.0 + k * k));

        let a1 = -1.5 * c3 * (2.0 * a21 + a23 + 5.0 * d21) - 0.375 * c4 * (12.0 - k * k);
        let a2 = 1.5 * c3 * (a24 - 2.0 * a22) + 1.125 * c4;
        let denom = 2.0 * lambda * (lambda * (1.0 + k * k) - 2.0 * k);
        let s1 = (1.5 * c3 * (2.0 * a21 * (k * k - 2.0) - a23 * (k * k + 2.0) - 2.0 * k * b21)
            - 0.375 * c4 * (3.0 * ipow(k, 4) - 8.0 * k * k + 8.0))
            / denom;
        let s2 = (1.5
            * c3
            * (2.0 * a22 * (k * k - 2.0) + a24 * (k * k + 2.0) + 2.0 * k * b22 + 5.0 * d21)
            + 0.375 * c4 * (12.0 - k * k))
            / denom;
        let l1 = a1 + 2.0 * l2f * s1;
        let l2 = a2 + 2.0 * l2f * s2;

        Some(Self {
            gamma,
            c2,
            lambda,
            k,
            a21,
            a22,
            a23,
            a24,
            b21,
            b22,
            d21,
            a31,
            a32,
            b31,
            b32,
            d31,
            d32,
            s1,
            s2,
            l1,
            l2,
        })
    }

    /// The in-plane amplitude `A_x` slaved to `A_z` by the amplitude constraint
    /// `l1 A_x² + l2 A_z² + Δ = 0`, `Δ = λ² − c2`. `None` if the constraint has no real root
    /// (never for L1/L2, where `l1 < 0` and `Δ > 0`).
    pub(crate) fn ax_from_az(&self, az: f64) -> Option<f64> {
        let delta = self.lambda * self.lambda - self.c2;
        let ax_sq = -(self.l2 * az * az + delta) / self.l1;
        (ax_sq > 0.0).then(|| libm::sqrt(ax_sq))
    }
}

/// The perpendicular-crossing synodic state `[x0, 0, z0, 0, ẏ0, 0]` of the halo about collinear point
/// `li` (0 → L1, 1 → L2) with Richardson γ-normalized out-of-plane amplitude `az` (synodic
/// `z0 ≈ γ·az`). `northern` selects the class (`δ_n = +1` northern, `−1` southern). Feed the result to
/// [`crate::periodic::halo_orbit`] to converge the exact orbit; it is a third-order-accurate seed.
/// Returns `None` if `li > 1` or the amplitude is invalid.
#[must_use]
pub fn richardson_halo_guess(sys: Cr3bp, li: usize, az: f64, northern: bool) -> Option<[f64; 6]> {
    let co = Coefficients::new(sys, li)?;
    let ax = co.ax_from_az(az)?;
    let delta = if northern { 1.0 } else { -1.0 };
    let g = co.gamma;
    let x_li = sys.collinear_points()[li].x;
    let omega = co.lambda * (1.0 + co.s1 * ax * ax + co.s2 * az * az);

    // Richardson series at the crossing θ = 0 (cos = 1, sin = 0), in γ-scaled Li-centred coordinates.
    let x0 = co.a21 * ax * ax + co.a22 * az * az - ax
        + (co.a23 * ax * ax - co.a24 * az * az)
        + (co.a31 * ipow(ax, 3) - co.a32 * ax * az * az);
    let z0 = delta * (az - 2.0 * co.d21 * ax * az + (co.d32 * az * ax * ax - co.d31 * ipow(az, 3)));
    // ẏ at the crossing is ω·dY/dθ with dY/dθ = k A_x + 2(b21 A_x² − b22 A_z²) + 3(b31 A_x³ − b32 A_x A_z²).
    let ydot = co.k * ax
        + 2.0 * (co.b21 * ax * ax - co.b22 * az * az)
        + 3.0 * (co.b31 * ipow(ax, 3) - co.b32 * ax * az * az);

    // Map to the synodic frame: origin at L_i, scale γ; Richardson +x aligns with synodic +x (sx = +1,
    // verified by the periodicity-defect sweep in the derivation). Velocities scale by γ, time shared.
    Some([x_li + g * x0, 0.0, g * z0, 0.0, g * omega * ydot, 0.0])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unreadable_literal, clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// The Rust coefficients reproduce the Maxima oracle (`derivations/richardson_halo.mac`) to 1e-9
    /// for Earth-Moon L1 — a full cross-check of the harmonic-balance formulas proven there.
    #[test]
    fn coefficients_match_the_cas_oracle_l1() {
        let co = Coefficients::new(Cr3bp::EARTH_MOON, 0).unwrap();
        let close =
            |a: f64, b: f64, name: &str| assert!((a - b).abs() < 1e-9, "{name}: {a} vs {b}");
        close(co.gamma, 0.150934286228, "gamma");
        close(co.c2, 5.1475945155, "c2");
        close(
            c_n(Cr3bp::EARTH_MOON.mu, co.gamma, 0, 3),
            3.2468421867,
            "c3",
        );
        close(
            c_n(Cr3bp::EARTH_MOON.mu, co.gamma, 0, 4),
            3.5847296922,
            "c4",
        );
        close(co.lambda, 2.334385880330, "lambda");
        close(co.k, 3.586499260960, "k");
        close(co.a21, 2.341950966941, "a21");
        close(co.a23, -0.9573660341050, "a23");
        close(co.b21, -0.4831630737764, "b21");
        close(co.d21, -0.2979105539904, "d21");
        close(co.a31, 0.9109199003834, "a31");
        close(co.b31, 1.206602980792, "b31");
        close(co.d32, 0.4463079758350, "d32");
        close(co.s1, -1.749109673946, "s1");
        close(co.s2, 0.08277770152554, "s2");
        close(co.l1, -28.79766725973, "l1");
        close(co.l2, 2.413764477677, "l2");
    }

    /// Earth-Moon L2 coefficients against the same Maxima oracle (c3 < 0 there — a sign-handling check).
    #[test]
    fn coefficients_match_the_cas_oracle_l2() {
        let co = Coefficients::new(Cr3bp::EARTH_MOON, 1).unwrap();
        let close =
            |a: f64, b: f64, name: &str| assert!((a - b).abs() < 1e-9, "{name}: {a} vs {b}");
        close(co.gamma, 0.167832748100, "gamma");
        close(
            c_n(Cr3bp::EARTH_MOON.mu, co.gamma, 1, 3),
            -2.6593351947,
            "c3",
        );
        close(co.lambda, 1.862645865425, "lambda");
        close(co.k, 2.912604127276, "k");
        close(co.l1, -8.889506839411, "l1");
        close(co.l2, 1.287156978370, "l2");
    }

    /// The synodic guess matches the Maxima oracle's emitted state for Earth-Moon L1, Az = 0.05.
    #[test]
    fn guess_matches_oracle_state_l1() {
        let g = richardson_halo_guess(Cr3bp::EARTH_MOON, 0, 0.05, true).unwrap();
        let close =
            |a: f64, b: f64, name: &str| assert!((a - b).abs() < 1e-8, "{name}: {a} vs {b}");
        close(g[0], 0.823807542, "x0");
        close(g[2], 0.008047180, "z0");
        close(g[4], 0.125947553, "vy0");
    }

    /// End-to-end: the Richardson seed converges under the differential corrector to a genuine halo
    /// about both L1 and L2, northern and southern, with the perpendicular-crossing closed and the
    /// out-of-plane sign matching the requested class. This is the load-bearing verification — a wrong
    /// coefficient would leave the seed too far for the 2×2 Newton to close.
    #[test]
    fn seed_converges_to_a_halo() {
        for &li in &[0usize, 1usize] {
            for &northern in &[true, false] {
                let orbit = crate::periodic::richardson_halo(Cr3bp::EARTH_MOON, li, 0.2, northern)
                    .unwrap_or_else(|| {
                        panic!("no convergence at L{}, northern={northern}", li + 1)
                    });
                // A real periodic orbit: positive period, on the correct side, correct z sign.
                assert!(orbit.period > 1.0, "degenerate period {}", orbit.period);
                assert!(orbit.state[2].abs() > 1e-3, "no out-of-plane amplitude");
                assert_eq!(orbit.state[2] > 0.0, northern, "wrong class at L{}", li + 1);
                // The corrector holds z0 and zeroes ẋ, ż at the crossing; verify perpendicularity by
                // re-running the guess and confirming the corrected x0 moved only modestly (good seed).
                let g = richardson_halo_guess(Cr3bp::EARTH_MOON, li, 0.2, northern).unwrap();
                assert!(
                    (orbit.state[0] - g[0]).abs() < 0.05,
                    "seed x0 far from converged"
                );
                assert!(
                    (orbit.state[4] - g[4]).abs() < 0.05,
                    "seed vy0 far from converged"
                );
            }
        }
    }
    /// The amplitude constraint `l1 Ax² + l2 Az² + Δ = 0` is claimed to have a real root for every
    /// collinear point, so `ax_from_az` is claimed never to fail. The claim rests on three signs —
    /// `l1 < 0`, `l2 > 0`, `Δ > 0` — which make `Ax² = (l2 Az² + Δ)/|l1| ≥ Δ/|l1| > 0` for every real
    /// `Az`, with no dependence on `Az` at all. Sweep the mass ratio over fourteen decades and assert
    /// the three signs directly, then check the extremes of the `Az` range the signs are supposed to
    /// cover.
    #[test]
    fn the_amplitude_constraint_has_a_real_root_at_every_mass_ratio() {
        let mut min_ax_sq = f64::INFINITY;
        for e in 0..=140i32 {
            let mu = 0.5 * libm::pow(10.0, -f64::from(e) / 10.0);
            for li in 0..2 {
                let co = Coefficients::new(Cr3bp { mu }, li).expect("li is 0 or 1");
                let delta = co.lambda * co.lambda - co.c2;
                let at = format!("mu={mu:e} L{}", li + 1);
                assert!(co.l1 < 0.0, "{at}: l1 = {} is not negative", co.l1);
                assert!(co.l2 > 0.0, "{at}: l2 = {} is not positive", co.l2);
                assert!(delta > 0.0, "{at}: delta = {delta} is not positive");
                let ax = co.ax_from_az(0.0).expect("a real root at Az = 0");
                min_ax_sq = min_ax_sq.min(ax * ax);
            }
        }
        assert!(
            min_ax_sq > 1e-3,
            "the constraint nearly degenerated: {min_ax_sq}"
        );
        // Ends of the range those signs cover: zero, subnormal and overflowing `Az` all keep a root;
        // only a `Az` that is not a number has none.
        let co = Coefficients::new(Cr3bp::EARTH_MOON, 0).expect("L1");
        for az in [0.0, f64::MIN_POSITIVE, 1e300, f64::INFINITY] {
            assert!(co.ax_from_az(az).is_some(), "no root at az = {az:e}");
        }
        assert!(
            co.ax_from_az(f64::NAN).is_none(),
            "a root from a NaN amplitude"
        );
    }
}
