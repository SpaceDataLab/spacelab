//! Zonal Hansen kernels K₀^{-n-1,s} and their χ-derivatives.
//!
//! Ported from Orekit's `HansenZonalLinear`. Orekit precomputes, per `s`, a set of polynomials in χ
//! (the composed linear transformations of `HansenUtilities.generateZonalPolynomials`) and evaluates
//! them at run time. Those polynomials are a *factored* form of the three-term recurrence
//! Danielson 2.7.3-(6c) / Collins 4-242; this module evaluates that same recurrence directly at the
//! working value of χ, which is mathematically identical and far shorter. It is validated against
//! Orekit's kernels through the mean-element-rate gate.
//!
//! Recurrence, with `K(n) ≡ K₀^{-n-1,s}` and `mnm1 = -n-1`:
//! * roots (Danielson 2.7.3-(6a,b)): `K(s) = 0`, `K(s+1) = χ^(2s+1) / 2^s`;
//! * step (n ≥ s+2): `K(n) = a·χ²·K(n-1) + b·χ²·K(n-2)` with
//!   `a = (1-n)(3-2n)/d`, `b = -(1-n)(2-n)/d`, `d = (1-n-s)(1-n+s)`.
//!
//! The χ-derivative is the exact chain-rule derivative of that recurrence (equivalent to Orekit's
//! separate `mpvecDeriv` polynomials): `K'(s+1) = (2s+1)·χ^(2s)/2^s`, and for n ≥ s+2
//! `K'(n) = (2/χ)·K(n) + χ²·(a·K'(n-1) + b·K'(n-2))`.
//!
//! `derivations/hansen_zonal.mac` checks the recurrence against the kernel's definition rather than
//! against Orekit, which runs the same recurrence. The definition is the orbit average
//! `⟨(r/a)^(-n-1) cos sf⟩ / eˢ`, integrated exactly as a polynomial in `e`. The recurrence and its
//! derivative match it for `s ≤ 8` and ten degrees above each `s`.

use std::vec::Vec;

/// Raise `x` to a small non-negative integer power by repeated multiplication (deterministic,
/// avoids a transcendental `pow` for the `χ^(2s±1)` roots).
fn ipow(x: f64, n: u32) -> f64 {
    let mut acc = 1.0;
    for _ in 0..n {
        acc *= x;
    }
    acc
}

/// Zonal Hansen kernels for one `s`, evaluated at a fixed χ over the degrees `n = 0..=n_max`.
///
/// `value(n)` returns K₀^{-n-1,s} and `derivative(n)` returns dK₀^{-n-1,s}/dχ; only `n ≥ s` are
/// meaningful (lower degrees are left at zero). The one exception is `n = s = 0`: the root is 0,
/// where K₀^{-1,0} = ⟨a/r⟩ = 1. It is harmless because the only step that reads it, `n = 2`,
/// multiplies it by `b = 0`.
#[derive(Debug, Clone)]
pub struct HansenZonal {
    values: Vec<f64>,
    derivs: Vec<f64>,
}

impl HansenZonal {
    /// Compute all kernels and χ-derivatives for order `s` up to degree `n_max` at the given χ.
    #[must_use]
    pub fn new(n_max: usize, s: usize, chi: f64) -> Self {
        let mut values = std::vec![0.0; n_max + 1];
        let mut derivs = std::vec![0.0; n_max + 1];

        // Roots K(s) = 0, K(s+1) = χ^(2s+1) / 2^s [Danielson 2.7.3-(6a,b)].
        let twots = ipow(2.0, s as u32);
        let s_u32 = s as u32;
        if s < n_max {
            values[s + 1] = ipow(chi, 2 * s_u32 + 1) / twots;
            derivs[s + 1] = (2.0 * s as f64 + 1.0) * ipow(chi, 2 * s_u32) / twots;
        }

        let chi2 = chi * chi;
        for n in (s + 2)..=n_max {
            let nf = n as f64;
            let sf = s as f64;
            // Coefficients from Collins 4-242 with mnm1 = -n-1.
            let d = (1.0 - nf - sf) * (1.0 - nf + sf);
            let a = (1.0 - nf) * (3.0 - 2.0 * nf) / d;
            let b = -(1.0 - nf) * (2.0 - nf) / d;
            let az = a * chi2;
            let bz = b * chi2;
            values[n] = az * values[n - 1] + bz * values[n - 2];
            derivs[n] = 2.0 / chi * values[n] + chi2 * (a * derivs[n - 1] + b * derivs[n - 2]);
        }

        Self { values, derivs }
    }

    /// The Hansen kernel K₀^{-n-1,s}.
    #[must_use]
    pub fn value(&self, n: usize) -> f64 {
        self.values[n]
    }

    /// The χ-derivative dK₀^{-n-1,s}/dχ.
    #[must_use]
    pub fn derivative(&self, n: usize) -> f64 {
        self.derivs[n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// K₀^{-3,0} = χ³ = (1 − e²)^{-3/2} — the classic closed form (Danielson), a direct check of the
    /// recurrence root and one step.
    #[test]
    fn k0_minus3_0_is_chi_cubed() {
        let chi = 1.03;
        let h = HansenZonal::new(4, 0, chi);
        assert!((h.value(2) - chi * chi * chi).abs() < 1e-14);
    }

    /// K₀^{-3,0} derivative is 3χ² (d/dχ of χ³).
    #[test]
    fn k0_minus3_0_derivative_is_three_chi_squared() {
        let chi = 1.011;
        let h = HansenZonal::new(4, 0, chi);
        assert!((h.derivative(2) - 3.0 * chi * chi).abs() < 1e-12);
    }
}
