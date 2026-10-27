//! Scattering phase functions: what a single scattering event does to a photon's direction.
//!
//! # The normalisation, stated once
//!
//! Everything here works with `p(μ)`, the probability density of the cosine of the scattering angle,
//! normalised as
//!
//! ```text
//!     ∫₋₁¹ p(μ) dμ = 1.
//! ```
//!
//! Two other conventions are common and neither is used: the solid-angle density `p(μ)/2π`, for which
//! `∫ p dΩ = 1`, and the astronomers' `P(μ) = 2p(μ)`, for which `(1/4π) ∫ P dΩ = 1`. The tests state
//! the first of those explicitly so the convention is pinned by a check rather than by this comment.
//!
//! The asymmetry parameter is `g = ∫₋₁¹ μ p(μ) dμ` in every convention.
//!
//! # Why more than one member
//!
//! Henyey–Greenstein is a one-parameter analytic fit with a closed-form CDF. It is smooth by
//! construction, so there is no `g` at which it produces the 22° and 46° refraction peaks of
//! hexagonal ice, the backscatter glory of water droplets, or the sharp forward diffraction lobe of
//! coarse aerosol — it can only lean toward the forward hemisphere as a whole by pushing `g` to 1,
//! distorting everything else to do it. Over cirrus, which covers a large fraction of any optical
//! scene, that is the assumption in an atmospheric correction that fails first.
//!
//! [`PhaseFunction::Tabulated`] is the way out: any angular structure at all, sampled by exact
//! inversion of the CDF of the *interpolant*. See [`TabulatedPhase`] for what that trades.
//!
//! Gate: `derivations/phase_functions.mac` derives the Henyey–Greenstein CDF, its first moment and
//! its Legendre moments `gˡ`, the exact inverse of both samplers, the Rayleigh CDF and Cardano root,
//! and the segment moments the table uses; `derivations/phase_functions.gp` re-derives them by
//! quadrature. The module test `the_phase_functions_reproduce_their_derivation` (G2) holds the
//! shipped arithmetic to that file's 40-digit values.

// The grid's endpoints are a contract — `mu[0]` is exactly `−1` and `mu[n−1]` exactly `1`, because a
// tolerance there would silently accept a table that does not span the sphere — and the Legendre and
// interpolation arithmetic is written in the single letters of the textbooks it comes from.
#![allow(
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]

use crate::mie;

/// Why a phase function was rejected.
///
/// Each names a specific violated precondition. A phase function that is not a probability density
/// is not a modelling choice with worse accuracy — it is a photon budget that does not close, and
/// the Monte Carlo has no way to notice downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseError {
    /// A grid of fewer than two nodes, or grids of differing length.
    GridTooShort,
    /// The `μ` grid is not strictly increasing, or does not run from `−1` to `+1`.
    GridNotAscending,
    /// A negative or non-finite phase-function value: not a density.
    NotADensity,
    /// The values integrate to zero, so there is nothing to normalise.
    ZeroIntegral,
    /// A Henyey–Greenstein asymmetry outside `(−1, 1)`, or a mixing weight outside `[0, 1]`.
    ParameterOutOfRange,
}

/// A phase function sampled on a `μ` grid, linear in `μ` between nodes.
///
/// # What the interpolation costs
///
/// The analytic members invert their own CDF exactly. This one cannot: a table is only the values at
/// its nodes. The choice made here is to be exact about *something* — the piecewise-linear
/// interpolant of `p` has a piecewise-quadratic CDF, which is inverted in closed form, so sampling
/// introduces **no error beyond the interpolant itself**. The error is therefore the interpolation
/// error of `p`, `≈ h²|p''|/8` per interval, and it is under the caller's control through the grid:
/// refine near a peak and it falls quadratically. It is *not* a sampling bias that no amount of
/// photons removes, which is what inverting a spline-fitted CDF or searching a coarse cumulative
/// table would have given.
///
/// The grid must run from `μ = −1` to `μ = +1` so the density is complete; values are normalised at
/// construction, so a caller may supply an unnormalised cross-section.
#[derive(Clone, Copy, Debug)]
pub struct TabulatedPhase<'a> {
    mu: &'a [f64],
    value: &'a [f64],
    cumulative: &'a [f64],
}

impl<'a> TabulatedPhase<'a> {
    /// Normalise `value` in place to `∫ p dμ = 1` and fill `cumulative` with the CDF at the nodes.
    ///
    /// Split from [`new`](Self::new) because the borrow checker will not hold a shared and an
    /// exclusive reference to the same buffers at once, and because it makes the mutation explicit:
    /// this is the call that rescales the caller's cross-section.
    ///
    /// # Errors
    ///
    /// [`PhaseError::GridTooShort`], [`PhaseError::GridNotAscending`], [`PhaseError::NotADensity`]
    /// or [`PhaseError::ZeroIntegral`].
    pub fn prepare(
        mu: &[f64],
        value: &mut [f64],
        cumulative: &mut [f64],
    ) -> Result<(), PhaseError> {
        check_grid(mu, value)?;
        if cumulative.len() != mu.len() {
            return Err(PhaseError::GridTooShort);
        }

        let mut total = 0.0;
        for i in 0..mu.len() - 1 {
            total += 0.5 * (mu[i + 1] - mu[i]) * (value[i] + value[i + 1]);
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(PhaseError::ZeroIntegral);
        }
        for v in value.iter_mut() {
            *v /= total;
        }

        cumulative[0] = 0.0;
        for i in 0..mu.len() - 1 {
            cumulative[i + 1] =
                cumulative[i] + 0.5 * (mu[i + 1] - mu[i]) * (value[i] + value[i + 1]);
        }
        // The last node is 1 by construction of the normalisation; assigning it removes the rounding
        // that would otherwise leave a sliver of ξ above the top of the table.
        let last = cumulative.len() - 1;
        cumulative[last] = 1.0;
        Ok(())
    }

    /// Wrap grids already run through [`prepare`](Self::prepare).
    ///
    /// # Errors
    ///
    /// As [`prepare`](Self::prepare), plus [`PhaseError::ZeroIntegral`] if the cumulative table does
    /// not start at 0 and end at 1.
    pub fn new(mu: &'a [f64], value: &'a [f64], cumulative: &'a [f64]) -> Result<Self, PhaseError> {
        check_grid(mu, value)?;
        if cumulative.len() != mu.len() {
            return Err(PhaseError::GridTooShort);
        }
        let last = cumulative.len() - 1;
        if cumulative[0] != 0.0 || (cumulative[last] - 1.0).abs() > 1e-12 {
            return Err(PhaseError::ZeroIntegral);
        }
        Ok(Self {
            mu,
            value,
            cumulative,
        })
    }

    /// The interval index `i` with `mu[i] ≤ μ < mu[i+1]`, clamped to the last interval.
    fn interval(grid: &[f64], x: f64) -> usize {
        let (mut lo, mut hi) = (0usize, grid.len() - 1);
        while hi - lo > 1 {
            let mid = usize::midpoint(lo, hi);
            if grid[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn value_at(&self, mu: f64) -> f64 {
        let i = Self::interval(self.mu, mu);
        let h = self.mu[i + 1] - self.mu[i];
        let t = ((mu - self.mu[i]) / h).clamp(0.0, 1.0);
        self.value[i] + t * (self.value[i + 1] - self.value[i])
    }

    /// Invert the CDF of the interpolant. On `[μᵢ, μᵢ₊₁]` with `p` linear the CDF is
    /// `F(t) = Fᵢ + pᵢ t + (pᵢ₊₁ − pᵢ) t² / 2h`, a quadratic solved in closed form.
    fn sample_at(&self, xi: f64) -> f64 {
        let i = Self::interval(self.cumulative, xi.clamp(0.0, 1.0));
        let h = self.mu[i + 1] - self.mu[i];
        let (p0, p1) = (self.value[i], self.value[i + 1]);
        let d = xi - self.cumulative[i];
        let a = 0.5 * (p1 - p0) / h;
        // Stable root: `2d / (b + √(b² + 4ad))` tends to `d/b` as `a → 0` without cancelling, and
        // the discriminant is `≥ p₁²` wherever `d` is inside the interval, so it cannot go negative.
        let discriminant = libm::sqrt((p0 * p0 + 4.0 * a * d).max(0.0));
        let denominator = p0 + discriminant;
        let t = if denominator > 0.0 {
            2.0 * d / denominator
        } else {
            0.0
        };
        (self.mu[i] + t.clamp(0.0, h)).clamp(-1.0, 1.0)
    }

    /// `g = ∫ μ p dμ`, exact for the piecewise-linear interpolant this table stands for.
    fn asymmetry_of(&self) -> f64 {
        let mut g = 0.0;
        for i in 0..self.mu.len() - 1 {
            let (a, b) = (self.mu[i], self.mu[i + 1]);
            let (p0, p1) = (self.value[i], self.value[i + 1]);
            // ∫ₐᵇ μ (p₀ + (p₁−p₀)(μ−a)/h) dμ, expanded so the h in the denominator cancels.
            let h = b - a;
            g += h * (p0 * (2.0 * a + b) + p1 * (a + 2.0 * b)) / 6.0;
        }
        g
    }
}

fn check_grid(mu: &[f64], value: &[f64]) -> Result<(), PhaseError> {
    if mu.len() < 2 || mu.len() != value.len() {
        return Err(PhaseError::GridTooShort);
    }
    if mu[0] != -1.0 || mu[mu.len() - 1] != 1.0 {
        return Err(PhaseError::GridNotAscending);
    }
    if mu.windows(2).any(|w| w[1] <= w[0]) {
        return Err(PhaseError::GridNotAscending);
    }
    if value.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(PhaseError::NotADensity);
    }
    Ok(())
}

/// A single-scattering phase function.
///
/// Every member can be evaluated at a cosine and sampled from a uniform deviate; the two analytic
/// ones invert their own CDF exactly, and [`Tabulated`](Self::Tabulated) inverts the CDF of its
/// interpolant exactly. `Copy`, so it sits inside a `Copy` atmosphere without a heap allocation
/// anywhere — the table is borrowed from the caller.
#[derive(Clone, Copy, Debug)]
pub enum PhaseFunction<'a> {
    /// Henyey–Greenstein, `p(μ) = (1 − g²) / (2 (1 + g² − 2gμ)^{3/2})`.
    ///
    /// One parameter, closed-form CDF, no structure. The right choice when the asymmetry parameter
    /// is all that is known about the medium, and the wrong one whenever an angular feature matters.
    HenyeyGreenstein {
        /// `g ∈ (−1, 1)`: `0` isotropic, `> 0` forward-peaked.
        asymmetry: f64,
    },
    /// Two Henyey–Greenstein lobes, `w p(μ; g_f) + (1 − w) p(μ; g_b)`.
    ///
    /// The cheap middle ground: with `g_f > 0` and `g_b < 0` it represents a forward peak *and* a
    /// backscatter rise at once, which one lobe cannot. Still smooth, so still no halo — this buys
    /// the gross shape of a cloud phase function, not its features.
    DoubleHenyeyGreenstein {
        /// Asymmetry of the forward lobe, in `(−1, 1)`.
        forward: f64,
        /// Asymmetry of the backward lobe, in `(−1, 1)`.
        backward: f64,
        /// Weight of the forward lobe, in `[0, 1]`.
        forward_weight: f64,
    },
    /// Rayleigh, `p(μ) = ⅜(1 + μ²)` — molecular scattering, with no parameter at all.
    ///
    /// The one phase function in this enum that is derived rather than fitted: it is what an induced
    /// dipole radiates, for particles small against the wavelength. Symmetric, `g = 0`, and twice as
    /// bright forward and back as it is at right angles — which is why it cannot be spelled as a
    /// Henyey–Greenstein at `g = 0`, whose isotropic `p = ½` misses the `1 + μ²` shape entirely.
    Rayleigh,
    /// An arbitrary tabulated density — the member with angular structure.
    Tabulated(TabulatedPhase<'a>),
}

impl PhaseFunction<'_> {
    /// The density `p(μ)` with `∫₋₁¹ p dμ = 1`.
    #[must_use]
    pub fn value(&self, mu: f64) -> f64 {
        match *self {
            Self::HenyeyGreenstein { asymmetry } => henyey_greenstein_value(asymmetry, mu),
            Self::DoubleHenyeyGreenstein {
                forward,
                backward,
                forward_weight,
            } => {
                forward_weight * henyey_greenstein_value(forward, mu)
                    + (1.0 - forward_weight) * henyey_greenstein_value(backward, mu)
            }
            Self::Rayleigh => 0.375 * (1.0 + mu * mu),
            Self::Tabulated(table) => table.value_at(mu),
        }
    }

    /// Draw `μ` from the phase function given a uniform deviate `ξ ∈ [0, 1)`.
    ///
    /// The mixture is sampled by choosing a lobe and then inverting that lobe — exact, and it needs
    /// only the one deviate because the leading `forward_weight` of the unit interval selects the
    /// forward lobe and the rest is rescaled back to `[0, 1)`. Consuming a second deviate would work
    /// too and would change every existing photon stream for no gain.
    #[must_use]
    pub fn sample(&self, xi: f64) -> f64 {
        match *self {
            Self::HenyeyGreenstein { asymmetry } => sample_henyey_greenstein(asymmetry, xi),
            Self::DoubleHenyeyGreenstein {
                forward,
                backward,
                forward_weight,
            } => {
                if xi < forward_weight {
                    sample_henyey_greenstein(forward, xi / forward_weight)
                } else {
                    sample_henyey_greenstein(
                        backward,
                        (xi - forward_weight) / (1.0 - forward_weight),
                    )
                }
            }
            Self::Rayleigh => sample_rayleigh(xi),
            Self::Tabulated(table) => table.sample_at(xi),
        }
    }

    /// The asymmetry parameter `g = ∫ μ p dμ`.
    ///
    /// Analytic for the Henyey–Greenstein members — `g` is the parameter, and the mixture's is the
    /// weighted mean — and computed from the table for [`Tabulated`](Self::Tabulated), where it is
    /// the number a caller compares against a published one.
    #[must_use]
    pub fn asymmetry(&self) -> f64 {
        match *self {
            Self::HenyeyGreenstein { asymmetry } => asymmetry,
            Self::DoubleHenyeyGreenstein {
                forward,
                backward,
                forward_weight,
            } => forward_weight * forward + (1.0 - forward_weight) * backward,
            Self::Rayleigh => 0.0,
            Self::Tabulated(table) => table.asymmetry_of(),
        }
    }

    /// Whether the parameters are inside their domains.
    ///
    /// # Errors
    ///
    /// [`PhaseError::ParameterOutOfRange`].
    pub fn validate(&self) -> Result<(), PhaseError> {
        let ok = match *self {
            Self::HenyeyGreenstein { asymmetry } => asymmetry.abs() < 1.0,
            Self::DoubleHenyeyGreenstein {
                forward,
                backward,
                forward_weight,
            } => {
                forward.abs() < 1.0 && backward.abs() < 1.0 && (0.0..=1.0).contains(&forward_weight)
            }
            Self::Rayleigh | Self::Tabulated(_) => true,
        };
        if ok {
            Ok(())
        } else {
            Err(PhaseError::ParameterOutOfRange)
        }
    }
}

/// `p(μ) = (1 − g²) / (2 (1 + g² − 2gμ)^{3/2})`.
fn henyey_greenstein_value(g: f64, mu: f64) -> f64 {
    let denominator = 1.0 + g * g - 2.0 * g * mu;
    (1.0 - g * g) / (2.0 * denominator * libm::sqrt(denominator))
}

/// The exact inverse of the Henyey–Greenstein CDF.
fn sample_henyey_greenstein(g: f64, xi: f64) -> f64 {
    if g.abs() < 1e-6 {
        2.0 * xi - 1.0 // isotropic
    } else {
        let t = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi);
        ((1.0 + g * g - t * t) / (2.0 * g)).clamp(-1.0, 1.0)
    }
}

/// The exact inverse of the Rayleigh CDF.
///
/// `F(μ) = (μ³ + 3μ + 4)/8`, so sampling solves the depressed cubic `μ³ + 3μ + (4 − 8ξ) = 0`. Its
/// discriminant `(2 − 4ξ)² + 1` is strictly positive, so there is exactly one real root and Cardano
/// gives it without a trigonometric detour: with `u = 4ξ − 2` and `s = √(u² + 1)`,
/// `μ = ∛(u + s) + ∛(u − s)`. The second cube root is not evaluated — `(u + s)(u − s) = −1` makes it
/// `−1/∛(u + s)`, which is one `cbrt` instead of two and avoids the cube root of a negative
/// altogether. `u + s > 0` for every real `u`, so the remaining root is of a positive number and the
/// reciprocal cannot divide by zero.
///
/// For `u < 0` that sum is `−|u| + √(u² + 1)`, which cancels; the same identity read the other way,
/// `u + s = 1/(s − u)`, is a sum of positives there. Measured over 10 001 deviates the round trip
/// `F(sample(ξ)) − ξ` is 3.3e-16 worst-case with the branch and 9.7e-16 without it.
///
/// The endpoints are the residual: `ξ = 0` wants exactly `−1`, and `∂(a − 1/a)/∂a` is `1 + φ²` there,
/// so one ulp in the cube root is two in the answer. The clamp is what guarantees the *contract* —
/// `|μ| ≤ 1`, which is what a downstream `acos` needs — rather than the arithmetic.
fn sample_rayleigh(xi: f64) -> f64 {
    let u = 4.0 * xi.clamp(0.0, 1.0) - 2.0;
    let s = libm::sqrt(u * u + 1.0);
    let a = libm::cbrt(if u >= 0.0 { u + s } else { 1.0 / (s - u) });
    (a - 1.0 / a).clamp(-1.0, 1.0)
}

/// Tabulate a phase function from its Legendre moments, `p(μ) = Σ (2l+1)/2 · χ_l P_l(μ)`.
///
/// `moments[0]` must be 1 for a normalised density and `moments[1]` is the asymmetry parameter; the
/// values are normalised at construction regardless, so an expansion truncated mid-series still
/// yields a density. A Legendre expansion is the compact interchange format for a phase function —
/// what a doubling-adding or discrete-ordinates code consumes — but it has no invertible CDF, so it
/// is a *constructor* for [`TabulatedPhase`] rather than a variant with its own sampler.
///
/// Truncating an expansion can drive `p` negative between nodes; the negative part is clipped to
/// zero, which is the standard remedy and is why the result is renormalised afterwards. A caller who
/// needs the clipping to be small should either supply more moments or use delta-M scaling first.
///
/// # Errors
///
/// As [`TabulatedPhase::prepare`].
pub fn tabulate_legendre(
    moments: &[f64],
    mu: &[f64],
    value: &mut [f64],
    cumulative: &mut [f64],
) -> Result<(), PhaseError> {
    if mu.len() != value.len() {
        return Err(PhaseError::GridTooShort);
    }
    for (out, &x) in value.iter_mut().zip(mu.iter()) {
        // Bonnet's recursion, evaluated alongside the sum so no array of P_l is needed.
        let (mut previous, mut current) = (1.0, x);
        let mut sum = moments.first().copied().unwrap_or(1.0) * 0.5;
        for (l, &moment) in moments.iter().enumerate().skip(1) {
            let degree = l as f64;
            let two_l_plus_one = 2.0 * degree + 1.0;
            sum += two_l_plus_one * 0.5 * moment * current;
            let next = (two_l_plus_one * x * current - degree * previous) / (degree + 1.0);
            previous = current;
            current = next;
        }
        *out = sum.max(0.0);
    }
    TabulatedPhase::prepare(mu, value, cumulative)
}

/// Tabulate the Mie phase function of a size distribution of spheres onto `mu`.
///
/// A thin wrapper over [`mie::phase_on_grid`] that finishes with
/// [`TabulatedPhase::prepare`], so the caller gets a ready density in one call.
///
/// # Errors
///
/// As [`TabulatedPhase::prepare`]. A Mie computation that does not converge leaves the grid
/// untouched and surfaces as [`PhaseError::ZeroIntegral`].
pub fn tabulate_mie(
    distribution: &mie::SizeDistribution,
    refractive_index: mie::Complex,
    wavelength: f64,
    mu: &[f64],
    value: &mut [f64],
    cumulative: &mut [f64],
) -> Result<mie::BulkOptics, PhaseError> {
    let optics = mie::phase_on_grid(distribution, refractive_index, wavelength, mu, value)
        .ok_or(PhaseError::ZeroIntegral)?;
    TabulatedPhase::prepare(mu, value, cumulative)?;
    Ok(optics)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::unreadable_literal,
        clippy::excessive_precision,
        clippy::items_after_statements
    )]

    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// (G2) The densities and samplers reproduce, at 40-digit points, the definitions
    /// `derivations/phase_functions.mac` derives them from.
    ///
    /// That file shows that the coded HG CDF has `F' = p` and `F(±1) = 0, 1`, that the HG sampler is
    /// its exact inverse, and that Cardano's root `a − 1/a` solves the Rayleigh CDF's cubic. The
    /// points here are where those closed forms come out exact or algebraic, so the shipped
    /// arithmetic is held to the rounding of one evaluation rather than to a quadrature:
    /// `μ = 1147/1352` at `g = 3/4, ξ = 3/8`, `μ = −5/16` at `g = −5/8, ξ = 13/16`, and the real
    /// root of `μ³ + 3μ + 4 − 8ξ` at three deviates on both sides of the branch at `u = 0`.
    /// `derivations/phase_functions.gp` re-derives each literal by quadrature and `polroots`. The
    /// tabulated member is pinned through its exact first moment: the five-node ramp `k/6` has
    /// `g = 2/9`.
    #[test]
    fn the_phase_functions_reproduce_their_derivation() {
        let hg = PhaseFunction::HenyeyGreenstein { asymmetry: 0.75 };
        let close = |got: f64, want: f64, what: &str| {
            assert!(
                (got - want).abs() <= 4.0 * f64::EPSILON * want.abs().max(0.25),
                "{what}: {got:e} against {want:e}"
            );
        };
        close(
            hg.value(0.5),
            2.986847210443541426252609333999227411096e-1,
            "HG p(1/2) at g = 3/4",
        );
        close(
            hg.sample(0.375),
            8.483727810650887573964497041420118343195e-1,
            "HG μ(3/8) at g = 3/4, 1147/1352",
        );
        close(
            PhaseFunction::HenyeyGreenstein { asymmetry: -0.625 }.sample(0.8125),
            -0.3125,
            "HG μ(13/16) at g = −5/8",
        );
        for (xi, want) in [
            (0.25, -5.960716379833215231128054143996818281133e-1),
            (0.3125, -4.662205239107734273913577876420742203745e-1),
            (0.875, 8.177316738868235060940870704093553698711e-1),
        ] {
            close(PhaseFunction::Rayleigh.sample(xi), want, "Rayleigh μ(ξ)");
        }

        let mu = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let mut value = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut cumulative = [0.0; 5];
        TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).unwrap();
        let table =
            PhaseFunction::Tabulated(TabulatedPhase::new(&mu, &value, &cumulative).unwrap());
        close(
            table.asymmetry(),
            2.222222222222222222222222222222222222222e-1,
            "table g, 2/9",
        );
    }

    /// An angle grid fine enough to resolve a halo: 0.25° in scattering angle.
    fn angle_grid(nodes: usize) -> Vec<f64> {
        (0..nodes)
            .map(|i| {
                let theta = core::f64::consts::PI * (nodes - 1 - i) as f64 / (nodes - 1) as f64;
                libm::cos(theta).clamp(-1.0, 1.0)
            })
            .collect()
    }

    /// `∫₋₁¹ f dμ` by Simpson's rule on `2n + 1` points — for the analytic members, whose `p` is
    /// smooth, this is exact to well past the tolerances below.
    fn simpson(n: usize, f: impl Fn(f64) -> f64) -> f64 {
        let nodes = 2 * n + 1;
        let h = 2.0 / (nodes - 1) as f64;
        let mut total = 0.0;
        for i in 0..nodes {
            let weight = if i == 0 || i == nodes - 1 {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            total += weight * f(-1.0 + h * i as f64);
        }
        total * h / 3.0
    }

    /// Simpson's rule per interval of the table's own grid. The integrand `w(μ) p(μ)` is at most
    /// quadratic there for a linear `w` and a piecewise-linear `p`, so this is *exact* for the
    /// interpolant the table stands for — a residual is a fault, not a quadrature error. Simpson
    /// rather than the trapezoid precisely because `μ p(μ)` is quadratic and the trapezoid is not
    /// exact on it: at 721 nodes that alone shifts `g` by 1.2e-6, which is the wrong thing to be
    /// measuring.
    fn table_integral(mu: &[f64], value: &[f64], weight: impl Fn(f64) -> f64) -> f64 {
        let mut total = 0.0;
        for i in 0..mu.len() - 1 {
            let h = mu[i + 1] - mu[i];
            let middle = f64::midpoint(mu[i], mu[i + 1]);
            let value_middle = f64::midpoint(value[i], value[i + 1]);
            total += h / 6.0
                * (weight(mu[i]) * value[i]
                    + 4.0 * weight(middle) * value_middle
                    + weight(mu[i + 1]) * value[i + 1]);
        }
        total
    }

    /// (G4) Every member is a probability density on the sphere: `∫ p dΩ = 1`.
    ///
    /// Written in the solid-angle form the ticket states, `2π ∫₋₁¹ p(μ)/2π dμ`, so the check pins the
    /// convention and not just the normalisation constant — a member that integrated to `4π` or to
    /// `1/2` would still look self-consistent inside the sampler and would silently rescale every
    /// scattering event.
    #[test]
    fn every_member_integrates_to_one_over_the_sphere() {
        for g in [-0.8, -0.3, 0.0, 0.3, 0.85, 0.95] {
            let p = PhaseFunction::HenyeyGreenstein { asymmetry: g };
            let total = core::f64::consts::TAU
                * simpson(200_000, |mu| p.value(mu) / core::f64::consts::TAU);
            assert!((total - 1.0).abs() < 1e-9, "HG g={g}: ∫p dΩ = {total}");
        }
        let mixture = PhaseFunction::DoubleHenyeyGreenstein {
            forward: 0.9,
            backward: -0.5,
            forward_weight: 0.7,
        };
        let total = core::f64::consts::TAU
            * simpson(200_000, |mu| mixture.value(mu) / core::f64::consts::TAU);
        assert!((total - 1.0).abs() < 1e-9, "double HG: ∫p dΩ = {total}");
        let rayleigh = PhaseFunction::Rayleigh;
        let total = core::f64::consts::TAU
            * simpson(200_000, |mu| rayleigh.value(mu) / core::f64::consts::TAU);
        assert!((total - 1.0).abs() < 1e-12, "Rayleigh: ∫p dΩ = {total}");
    }

    /// Rayleigh is the shape a dipole radiates, and it is not the isotropic function.
    ///
    /// `p(±1)/p(0) = 2` exactly — forward and back are twice as bright as right angles — and `g = 0`
    /// despite that structure, because the lobes are symmetric. The second half is what makes the
    /// first worth asserting: an asymmetry parameter of zero is the whole of what a
    /// Henyey–Greenstein can say about this density, and it says nothing.
    #[test]
    fn the_rayleigh_density_is_twice_as_bright_along_the_beam_as_across_it() {
        let p = PhaseFunction::Rayleigh;
        assert_eq!(p.value(0.0), 0.375);
        assert_eq!(p.value(1.0), 0.75);
        assert_eq!(p.value(-1.0), 0.75);
        assert_eq!(p.asymmetry(), 0.0);
        assert!(
            simpson(200_000, |mu| mu * p.value(mu)).abs() < 1e-12,
            "the density's own g is not zero"
        );
        let isotropic = PhaseFunction::HenyeyGreenstein { asymmetry: 0.0 };
        assert!(
            (p.value(1.0) - isotropic.value(1.0)).abs() > 0.24,
            "an isotropic phase function is not a Rayleigh one"
        );
    }

    /// The Rayleigh sampler inverts its own CDF exactly, endpoints included.
    ///
    /// `F(μ) = (μ³ + 3μ + 4)/8`, and Cardano's root of the cubic it inverts is exact rather than
    /// iterative, so the round trip closes to a few ulp everywhere rather than to a solver
    /// tolerance. The endpoints are the case that would show a wrong branch: `ξ = 0` must give
    /// exactly −1 and `ξ = 1` exactly `+1`, and both come out of `a − 1/a` at the golden ratio.
    #[test]
    fn the_rayleigh_sampler_inverts_its_cdf() {
        let cdf = |mu: f64| (mu * mu * mu + 3.0 * mu + 4.0) / 8.0;
        for i in 0..=1000 {
            let xi = f64::from(i) / 1000.0;
            let mu = PhaseFunction::Rayleigh.sample(xi);
            assert!((-1.0..=1.0).contains(&mu), "ξ={xi} sampled {mu}");
            assert!(
                (cdf(mu) - xi).abs() < 1e-14,
                "ξ={xi}: F({mu}) = {}",
                cdf(mu)
            );
        }
        // The two ends are two ulp rather than exact, for the reason `sample_rayleigh` records; the
        // clamp is what makes them *legal*, and that is the part a caller depends on.
        assert!((PhaseFunction::Rayleigh.sample(0.0) + 1.0).abs() < 1e-15);
        assert_eq!(PhaseFunction::Rayleigh.sample(1.0), 1.0);
        assert_eq!(PhaseFunction::Rayleigh.sample(0.5), 0.0);
    }

    /// (G3) Every member reports the asymmetry parameter its own density has: `g = ∫ μ p dμ`.
    #[test]
    fn every_member_reports_the_asymmetry_of_its_own_density() {
        for g in [-0.8, -0.3, 0.0, 0.3, 0.85, 0.95] {
            let p = PhaseFunction::HenyeyGreenstein { asymmetry: g };
            let integral = simpson(200_000, |mu| mu * p.value(mu));
            assert!(
                (integral - p.asymmetry()).abs() < 1e-9,
                "HG g={g}: {integral}"
            );
        }
        let mixture = PhaseFunction::DoubleHenyeyGreenstein {
            forward: 0.9,
            backward: -0.5,
            forward_weight: 0.7,
        };
        let integral = simpson(200_000, |mu| mu * mixture.value(mu));
        assert!(
            (integral - mixture.asymmetry()).abs() < 1e-9,
            "double HG: {integral} vs {}",
            mixture.asymmetry()
        );
    }

    /// The Henyey–Greenstein sampler is the exact inverse of the Henyey–Greenstein CDF.
    ///
    /// The slab tests are statistical to three digits at best, and the ones at `g = 0` leave the
    /// whole non-isotropic arm dead — replacing the sampler by the constant `0.0` survived them. The
    /// phase function has a closed-form CDF (verified against `p` by differentiation in Maxima),
    ///
    ///     p(μ) = (1 − g²) / (2 (1 + g² − 2gμ)^{3/2}),
    ///     F(μ) = (1 − g²)/(2g) · [ (1 + g² − 2gμ)^{−1/2} − (1 + g)^{−1} ],
    ///
    /// and stating `F(sample(ξ)) = ξ` fixes the sampler without restating it: the CDF is the
    /// integral of the phase function, not a rearrangement of the sampling formula.
    #[test]
    fn the_henyey_greenstein_sampler_inverts_its_cdf() {
        fn cdf(g: f64, mu: f64) -> f64 {
            (1.0 - g * g) / (2.0 * g)
                * (1.0 / libm::sqrt(1.0 + g * g - 2.0 * g * mu) - 1.0 / (1.0 + g))
        }
        for g in [-0.95, -0.6, -0.1, 0.1, 0.5, 0.9, 0.95] {
            for k in 1..40 {
                let xi = f64::from(k) / 40.0;
                let mu = sample_henyey_greenstein(g, xi);
                assert!((cdf(g, mu) - xi).abs() < 1e-12, "g={g} ξ={xi}: μ={mu}");
            }
            // The two ends of the interval map to the two ends of the cosine. Exact in the reals;
            // in doubles the ratio `(1 − g²)/(1 ∓ g)` loses a few bits near |g| = 1.
            assert!(
                (sample_henyey_greenstein(g, 0.0) + 1.0).abs() < 1e-14,
                "g={g} ξ=0"
            );
            assert!(
                (sample_henyey_greenstein(g, 1.0) - 1.0).abs() < 1e-14,
                "g={g} ξ=1"
            );
        }
    }

    /// Sampled cosines average to `g` — the definition of the asymmetry parameter. A midpoint rule
    /// over ξ states that to seven digits, which no rearrangement of the formula survives.
    #[test]
    fn the_sampled_cosines_average_to_the_asymmetry_parameter() {
        const N: u32 = 20_000;
        for g in [-0.6, -0.1, 0.0, 0.1, 0.5, 0.9] {
            let mean = (0..N)
                .map(|k| sample_henyey_greenstein(g, (f64::from(k) + 0.5) / f64::from(N)))
                .sum::<f64>()
                / f64::from(N);
            assert!((mean - g).abs() < 1e-7, "g={g}: ⟨μ⟩ = {mean}");
        }
    }

    /// The isotropic shortcut is taken strictly below its threshold. At `|g| = 1e-6` exactly the two
    /// arms differ by 1.5e-6 in the sampled cosine — six orders below anything a photon count can
    /// resolve, but a difference all the same, and the boundary is the only input at which the
    /// guard's `<` and `<=` disagree.
    #[test]
    fn the_isotropic_shortcut_stops_short_of_its_threshold() {
        assert!(sample_henyey_greenstein(1e-7, 0.5).abs() < f64::EPSILON);
        let mu = sample_henyey_greenstein(1e-6, 0.5);
        assert!((mu - 1.5e-6).abs() < 1e-9, "g at the threshold: μ = {mu}");
    }

    /// The mixture is sampled from the mixture, not from either lobe: the empirical CDF of the
    /// one-deviate scheme matches `∫p` of the two-lobe density.
    #[test]
    fn the_two_lobe_sampler_draws_from_the_two_lobe_density() {
        let mixture = PhaseFunction::DoubleHenyeyGreenstein {
            forward: 0.85,
            backward: -0.6,
            forward_weight: 0.6,
        };
        const N: u32 = 200_000;
        for cut in [-0.9, -0.5, 0.0, 0.5, 0.9] {
            let below = (0..N)
                .filter(|k| mixture.sample((f64::from(*k) + 0.5) / f64::from(N)) < cut)
                .count() as f64
                / f64::from(N);
            let want = simpson(100_000, |mu| if mu < cut { mixture.value(mu) } else { 0.0 });
            assert!((below - want).abs() < 2e-3, "cut {cut}: {below} vs {want}");
        }
    }

    /// The tabulated sampler inverts the CDF of its own interpolant, exactly.
    ///
    /// This is the claim the [`TabulatedPhase`] doc makes and the one that separates it from a
    /// coarse cumulative search: `∫₋₁^{sample(ξ)} p dμ = ξ` to rounding, for a `p` that is *not* the
    /// analytic one — so no analytic identity can be hiding in it.
    #[test]
    fn the_tabulated_sampler_inverts_the_cdf_of_its_interpolant() {
        let mu = angle_grid(361);
        // A deliberately structured density: a narrow spike on a smooth background, which is what a
        // halo looks like and what a smooth sampler would get wrong.
        let mut value: Vec<f64> = mu
            .iter()
            .map(|m| 1.0 + 4.0 * libm::exp(-400.0 * (m - 0.927) * (m - 0.927)) + 0.5 * m)
            .collect();
        let mut cumulative = vec![0.0; mu.len()];
        TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).unwrap();
        let table = TabulatedPhase::new(&mu, &value, &cumulative).unwrap();

        for k in 1..200 {
            let xi = f64::from(k) / 200.0;
            let sampled = table.sample_at(xi);
            let mut integral = 0.0;
            for i in 0..mu.len() - 1 {
                if mu[i + 1] <= sampled {
                    integral += 0.5 * (mu[i + 1] - mu[i]) * (value[i] + value[i + 1]);
                } else if mu[i] < sampled {
                    let h = mu[i + 1] - mu[i];
                    let t = sampled - mu[i];
                    let end = value[i] + (value[i + 1] - value[i]) * t / h;
                    integral += 0.5 * t * (value[i] + end);
                    break;
                }
            }
            assert!((integral - xi).abs() < 1e-12, "ξ={xi}: ∫p = {integral}");
        }
    }

    /// A tabulated Henyey–Greenstein converges on the analytic one at the interpolant's rate.
    ///
    /// The point is not that a fine table is accurate — it is that the error is *interpolation*
    /// error and therefore falls as `h²`, which is the property that makes the grid a knob rather
    /// than a hazard. Measured here: quadrupling the node count divides the worst sampled-cosine
    /// error by close to four.
    #[test]
    fn the_tabulated_phase_function_converges_quadratically_on_the_analytic_one() {
        let analytic = PhaseFunction::HenyeyGreenstein { asymmetry: 0.7 };
        let mut errors = [0.0; 2];
        for (slot, nodes) in errors.iter_mut().zip([181usize, 721]) {
            let mu = angle_grid(nodes);
            let mut value: Vec<f64> = mu.iter().map(|m| analytic.value(*m)).collect();
            let mut cumulative = vec![0.0; nodes];
            TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).unwrap();
            let table =
                PhaseFunction::Tabulated(TabulatedPhase::new(&mu, &value, &cumulative).unwrap());
            let mut worst: f64 = 0.0;
            for k in 1..200 {
                let xi = f64::from(k) / 200.0;
                worst = worst.max((table.sample(xi) - analytic.sample(xi)).abs());
            }
            *slot = worst;
        }
        let ratio = errors[0] / errors[1];
        assert!(
            (8.0..24.0).contains(&ratio),
            "quadrupling the grid changed the error by {ratio}× (errors {errors:?}), not the ~16× of an h² interpolant"
        );
    }

    /// (G1) A Legendre expansion of a Henyey–Greenstein reproduces it — the moments of HG are `gˡ`, so
    /// this is an identity the constructor either satisfies or does not, with no fitting involved.
    #[test]
    fn a_legendre_expansion_reproduces_the_function_whose_moments_it_holds() {
        let g = 0.5;
        let moments: Vec<f64> = (0..48).map(|l| libm::pow(g, l as f64)).collect();
        let mu = angle_grid(721);
        let mut value = vec![0.0; mu.len()];
        let mut cumulative = vec![0.0; mu.len()];
        tabulate_legendre(&moments, &mu, &mut value, &mut cumulative).unwrap();
        let table = TabulatedPhase::new(&mu, &value, &cumulative).unwrap();
        let analytic = PhaseFunction::HenyeyGreenstein { asymmetry: g };
        assert!(
            (table.asymmetry_of() - g).abs() < 5e-3,
            "Legendre g = {}",
            table.asymmetry_of()
        );
        for m in [-0.9, -0.4, 0.0, 0.4, 0.9] {
            let (got, want) = (table.value_at(m), analytic.value(m));
            assert!(
                (got - want).abs() < 5e-3 * want.max(1.0),
                "μ={m}: {got} vs {want}"
            );
        }
    }

    /// A table integrates to one and reports its own `g` — the two invariants, on a density with no
    /// analytic form at all.
    #[test]
    fn a_tabulated_density_satisfies_both_invariants() {
        let mu = angle_grid(721);
        let mut value: Vec<f64> = mu
            .iter()
            .map(|m| 1.0 + 9.0 * libm::exp(-900.0 * (m - 0.927) * (m - 0.927)))
            .collect();
        let mut cumulative = vec![0.0; mu.len()];
        TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).unwrap();
        let table = TabulatedPhase::new(&mu, &value, &cumulative).unwrap();

        let total =
            core::f64::consts::TAU * table_integral(&mu, &value, |_| 1.0) / core::f64::consts::TAU;
        assert!((total - 1.0).abs() < 1e-14, "∫p dΩ = {total}");
        let g = table_integral(&mu, &value, |m| m);
        assert!(
            (g - table.asymmetry_of()).abs() < 1e-14,
            "{g} vs {}",
            table.asymmetry_of()
        );
    }

    /// Every rejection names its own precondition, and each is reachable.
    #[test]
    fn a_table_that_is_not_a_density_is_refused_by_name() {
        let mut value = [1.0, 1.0, 1.0];
        let mut cumulative = [0.0; 3];
        assert_eq!(
            TabulatedPhase::prepare(&[-1.0], &mut value[..1], &mut cumulative[..1]),
            Err(PhaseError::GridTooShort)
        );
        assert_eq!(
            TabulatedPhase::prepare(&[-1.0, 0.0, 1.0], &mut value, &mut cumulative[..2]),
            Err(PhaseError::GridTooShort)
        );
        assert_eq!(
            TabulatedPhase::prepare(&[-0.5, 0.0, 1.0], &mut value, &mut cumulative),
            Err(PhaseError::GridNotAscending)
        );
        assert_eq!(
            TabulatedPhase::prepare(&[-1.0, 0.5, 0.5], &mut value, &mut cumulative),
            Err(PhaseError::GridNotAscending)
        );
        let mut negative = [1.0, -1.0, 1.0];
        assert_eq!(
            TabulatedPhase::prepare(&[-1.0, 0.0, 1.0], &mut negative, &mut cumulative),
            Err(PhaseError::NotADensity)
        );
        let mut zero = [0.0, 0.0, 0.0];
        assert_eq!(
            TabulatedPhase::prepare(&[-1.0, 0.0, 1.0], &mut zero, &mut cumulative),
            Err(PhaseError::ZeroIntegral)
        );
        assert_eq!(
            TabulatedPhase::new(&[-1.0, 0.0, 1.0], &[1.0, 1.0, 1.0], &[0.0, 0.3, 0.7]).err(),
            Some(PhaseError::ZeroIntegral)
        );
        assert_eq!(
            PhaseFunction::HenyeyGreenstein { asymmetry: 1.0 }.validate(),
            Err(PhaseError::ParameterOutOfRange)
        );
        assert_eq!(
            PhaseFunction::DoubleHenyeyGreenstein {
                forward: 0.5,
                backward: -0.5,
                forward_weight: 1.5,
            }
            .validate(),
            Err(PhaseError::ParameterOutOfRange)
        );
        assert_eq!(
            PhaseFunction::HenyeyGreenstein { asymmetry: 0.99 }.validate(),
            Ok(())
        );
    }

    /// Both endpoints of a table are inside it.
    ///
    /// `interval` clamps to the last interval by construction, and that clamp is the whole reason
    /// `value_at` may index `i + 1` without checking. Evaluating at `μ = 1` and sampling at `ξ = 1`
    /// are the two ways to reach the top of the grid; if the search ever returned the last *node*
    /// instead of the last *interval*, both would read one past the end of the table.
    #[test]
    fn the_last_interval_holds_both_of_its_endpoints() {
        let mu = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let mut value = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut cumulative = [0.0; 5];
        TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).unwrap();
        let p = PhaseFunction::Tabulated(TabulatedPhase::new(&mu, &value, &cumulative).unwrap());
        assert!((p.value(1.0) - value[4]).abs() < 1e-15);
        assert!((p.value(-1.0) - value[0]).abs() < 1e-15);
        assert!((p.sample(1.0) - 1.0).abs() < 1e-9);
        assert!((p.sample(0.0) + 1.0).abs() < 1e-15);
    }

    /// A knot value is not enough to pin the interpolant; a point between two knots is.
    ///
    /// Everything else asked of a table integrates or normalises it, and both average the
    /// interpolation parameter away — a `t` that reads the wrong knot, or reads the interval
    /// backwards, still integrates to one. These two points are placed so that neither the affine map
    /// onto `[0, 1]` nor its reflection about the middle of the interval reproduces the other.
    #[test]
    fn the_table_interpolates_linearly_between_its_knots() {
        let mu = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let mut value = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut cumulative = [0.0; 5];
        TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).unwrap();
        let p = PhaseFunction::Tabulated(TabulatedPhase::new(&mu, &value, &cumulative).unwrap());
        // The trapezoid integral of the raw table is 6, so the normalised knots are `k / 6`.
        assert!((p.value(-0.25) - 2.5 / 6.0).abs() < 1e-15);
        assert!((p.value(0.625) - 4.25 / 6.0).abs() < 1e-15);
    }

    /// A knot where the density is zero samples to the knot, not to a NaN.
    ///
    /// The closed-form CDF inversion divides by `p₀ + √(p₀² + 4ad)`, and both terms vanish together
    /// exactly when the density is zero at the left knot and ξ lands on that knot. The guard returns
    /// the knot; without it the draw is `0/0`, and a NaN scattering direction propagates through the
    /// rest of the path silently instead of failing where it was made.
    #[test]
    fn a_knot_with_no_density_samples_to_the_knot_and_not_to_a_nan() {
        // Zero over the whole backward hemisphere, rising linearly to the forward pole. The integral
        // is already 1, so `prepare` leaves the values alone and the first two cumulative nodes
        // coincide — which is what puts ξ = 0 at the second knot with `p₀ = 0`.
        let mu = [-1.0, 0.0, 1.0];
        let mut value = [0.0, 0.0, 2.0];
        let mut cumulative = [0.0; 3];
        TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).unwrap();
        assert!(cumulative[1] <= 0.0, "ξ = 0 has to land on the second knot");
        let p = PhaseFunction::Tabulated(TabulatedPhase::new(&mu, &value, &cumulative).unwrap());
        assert!(p.sample(0.0).abs() < 1e-15, "sampled {}", p.sample(0.0));
    }

    /// Two nodes is a grid.
    ///
    /// `[-1, 1]` with a constant density is the isotropic table: one interval, one slope, and the
    /// shortest input the interpolation has anything to say about. It is also the shape a caller
    /// reaches for to express "no angular structure" in the same type as everything else.
    #[test]
    fn the_shortest_legal_grid_is_a_single_interval() {
        let mu = [-1.0, 1.0];
        let mut value = [1.0, 1.0];
        let mut cumulative = [0.0; 2];
        TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).unwrap();
        let p = PhaseFunction::Tabulated(TabulatedPhase::new(&mu, &value, &cumulative).unwrap());
        assert!((p.value(0.3) - 0.5).abs() < 1e-15);
        assert!(p.asymmetry().abs() < 1e-15);
        assert!(p.sample(0.5).abs() < 1e-15);
    }

    /// The split between the two lobes of a mixture belongs to the backward one.
    ///
    /// `sample` cuts the unit interval at `forward_weight` and rescales each side back to `[0, 1)`,
    /// so a deviate exactly at the cut is the backward lobe's ξ = 0 — its backscatter pole, μ = −1.
    /// Handing it to the forward lobe instead makes it that lobe's ξ = 1, μ = +1: the opposite pole,
    /// one ulp away from a deviate that samples correctly.
    #[test]
    fn the_split_between_the_lobes_belongs_to_the_backward_one() {
        let p = PhaseFunction::DoubleHenyeyGreenstein {
            forward: 0.85,
            backward: -0.6,
            forward_weight: 0.7,
        };
        assert!(
            (p.sample(0.7) + 1.0).abs() < 1e-15,
            "sampled {}",
            p.sample(0.7)
        );
        let below = p.sample(0.7 - f64::EPSILON);
        assert!((below - 1.0).abs() < 1e-9, "just below the split: {below}");
    }

    /// A mixture validates both of its lobes, not only its weight.
    ///
    /// The mixture arm is three conjuncts and the existing refusal test reaches only the weight. The
    /// boundary is the case worth pinning: at `|g| = 1` exactly the Henyey–Greenstein density has a
    /// zero denominator at one pole, so the domain is open and the endpoint has to be refused rather
    /// than clamped.
    #[test]
    fn a_mixture_validates_both_of_its_lobes() {
        fn mixture(forward: f64, backward: f64) -> PhaseFunction<'static> {
            PhaseFunction::DoubleHenyeyGreenstein {
                forward,
                backward,
                forward_weight: 0.7,
            }
        }
        assert_eq!(mixture(0.85, -0.6).validate(), Ok(()));
        for g in [1.0, -1.0, 1.5] {
            assert_eq!(
                mixture(g, -0.6).validate(),
                Err(PhaseError::ParameterOutOfRange),
                "forward {g}"
            );
            assert_eq!(
                mixture(0.85, g).validate(),
                Err(PhaseError::ParameterOutOfRange),
                "backward {g}"
            );
        }
    }
}
