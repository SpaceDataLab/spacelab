//! Mie scattering by a size distribution of homogeneous spheres.
//!
//! Exact solution of Maxwell's equations for a sphere, so it produces the angular structure a
//! one-parameter analytic fit cannot: the sharp forward diffraction lobe of coarse aerosol, the
//! rainbow near 140°, and the backscatter glory of water droplets. This is the member of
//! [`PhaseFunction`](crate::phase::PhaseFunction) for anything spherical — droplets, and aerosol
//! that is close enough to spherical to be worth the pretence.
//!
//! # The series, and where it is truncated
//!
//! For size parameter `x = 2πr/λ` and relative refractive index `m` the scattering coefficients are
//!
//! ```text
//!     aₙ = [(Dₙ/m + n/x) ψₙ − ψₙ₋₁] / [(Dₙ/m + n/x) ξₙ − ξₙ₋₁],
//!     bₙ = [(m Dₙ + n/x) ψₙ − ψₙ₋₁] / [(m Dₙ + n/x) ξₙ − ξₙ₋₁],
//! ```
//!
//! with `Dₙ = ψ'ₙ(mx)/ψₙ(mx)` the logarithmic derivative. Three numerical facts decide whether an
//! implementation works at all, and all three are followed here:
//!
//! * `Dₙ` is computed by **downward** recurrence from well above the truncation order. Upward is
//!   unstable and diverges for absorbing spheres — this is the single most common way a hand-rolled
//!   Mie code produces plausible nonsense.
//! * The series is truncated at Wiscombe's `nₘₐₓ = x + 4x^{1/3} + 2`, which is where the terms have
//!   fallen below `f64` significance rather than an arbitrary round number.
//! * `ψₙ` and `χₙ` go **upward**, which is stable for them.
//!
//! # Bulk quantities
//!
//! `Qₑₓₜ` and `Q_sca` come from the standard sums, and the asymmetry parameter from the closed form
//!
//! ```text
//!     g Q_sca = (4/x²) [ Σ n(n+2)/(n+1) Re(aₙ a*ₙ₊₁ + bₙ b*ₙ₊₁) + Σ (2n+1)/(n(n+1)) Re(aₙ b*ₙ) ],
//! ```
//!
//! deliberately *not* by quadrature over the tabulated phase function. The two are then independent,
//! and their agreement is a real check on the angular grid rather than a restatement of it — which
//! is what `the_tabulated_asymmetry_reproduces_the_mie_series_value` measures.

// The Mie recurrences are written in their textbook notation (x, m, n, a, b, D, ψ, χ, ξ), index
// arithmetic on the series moves between `usize` and `f64` at every term, and the Wiscombe
// truncation order is a real-valued expression rounded down — non-negative because `x ≥ 0`.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::similar_names
)]

use alloc::vec;
use alloc::vec::Vec;

/// A complex number, in the one place the workspace needs one outside the signal crate.
///
/// Private arithmetic rather than a dependency: Mie needs six operations, and a `no_std` crate whose
/// selling point is that it has no dependencies should not acquire one for `mul`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part. For a refractive index this is the **absorption** part and is `≥ 0` in the
    /// `m = n + ik` convention used throughout.
    pub im: f64,
}

impl Complex {
    /// `re + i·im`.
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    const fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    const fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }

    const fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    const fn scale(self, k: f64) -> Self {
        Self::new(self.re * k, self.im * k)
    }

    fn div(self, other: Self) -> Self {
        let d = other.re * other.re + other.im * other.im;
        Self::new(
            (self.re * other.re + self.im * other.im) / d,
            (self.im * other.re - self.re * other.im) / d,
        )
    }

    fn inverse(self) -> Self {
        Self::new(1.0, 0.0).div(self)
    }

    const fn conjugate(self) -> Self {
        Self::new(self.re, -self.im)
    }

    const fn norm_squared(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// A cloud or aerosol size distribution, normalised to unit number density.
///
/// Both members are parameterised by the **effective radius** `rₑ = ∫r³n dr / ∫r²n dr` rather than a
/// mode or median, because `rₑ` is the moment the optics actually depend on — two distributions with
/// the same `rₑ` and similar width scatter almost identically, which is Hansen and Travis's result
/// and the reason operational retrievals report `rₑ`.
#[derive(Clone, Copy, Debug)]
pub enum SizeDistribution {
    /// The Hansen–Travis modified gamma, `n(r) ∝ r^{(1−3v)/v} exp(−r/(rₑ v))` — the standard cloud
    /// droplet distribution.
    Gamma {
        /// Effective radius `rₑ`, in the same length unit as the wavelength.
        effective_radius: f64,
        /// Effective variance `v ∈ (0, 1/3)`. Water cloud is near `0.1`.
        effective_variance: f64,
    },
    /// Lognormal, `n(r) ∝ exp(−ln²(r/r₀) / 2σ²) / r` — the usual aerosol form.
    Lognormal {
        /// Median radius `r₀`, in the same length unit as the wavelength.
        median_radius: f64,
        /// Geometric standard deviation's logarithm `σ`. Continental aerosol is near `0.7`.
        log_sigma: f64,
    },
}

impl SizeDistribution {
    /// `n(r)`, up to a constant — the integrals here are all ratios, so the constant cancels.
    fn density(&self, r: f64) -> f64 {
        match *self {
            Self::Gamma {
                effective_radius,
                effective_variance,
            } => {
                let exponent = (1.0 - 3.0 * effective_variance) / effective_variance;
                libm::pow(r, exponent) * libm::exp(-r / (effective_radius * effective_variance))
            }
            Self::Lognormal {
                median_radius,
                log_sigma,
            } => {
                let t = libm::log(r / median_radius) / log_sigma;
                libm::exp(-0.5 * t * t) / r
            }
        }
    }

    /// The radius range the quadrature covers, chosen from the distribution's own scale so that the
    /// truncated tails carry a negligible share of the **third** moment — the slowest-converging one
    /// the optics need. Measured on both members at the settings the tests use, the omitted tails
    /// are below `1e-6` of `∫r³n dr`.
    fn radius_range(&self) -> (f64, f64) {
        match *self {
            Self::Gamma {
                effective_radius,
                effective_variance,
            } => (
                effective_radius * effective_variance * 0.02,
                effective_radius * (1.0 + 12.0 * effective_variance),
            ),
            Self::Lognormal {
                median_radius,
                log_sigma,
            } => (
                median_radius * libm::exp(-5.0 * log_sigma),
                median_radius * libm::exp(5.0 * log_sigma),
            ),
        }
    }
}

/// The size-integrated optical properties of a distribution, per particle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BulkOptics {
    /// Mean extinction cross-section, in squared wavelength units.
    pub extinction_cross_section: f64,
    /// Mean scattering cross-section, in squared wavelength units.
    pub scattering_cross_section: f64,
    /// Single-scattering albedo `ω = C_sca / C_ext`.
    pub single_scattering_albedo: f64,
    /// Asymmetry parameter, from the Mie series rather than from any angular grid.
    pub asymmetry: f64,
}

/// The quadrature nodes over radius. Simpson's rule on a uniform grid in `ln r`, which is the
/// natural variable for both distributions — one is a power law times an exponential, the other a
/// Gaussian in `ln r`, and both are smooth there. An odd count so Simpson's rule closes.
const RADIUS_NODES: usize = 129;

/// Series truncation, `nₘₐₓ = x + 4x^{1/3} + 2` (Wiscombe 1980).
fn truncation_order(x: f64) -> usize {
    (x + 4.0 * libm::cbrt(x) + 2.0) as usize + 1
}

/// The Mie coefficients `aₙ`, `bₙ` for one size parameter and refractive index.
///
/// Returns `None` for a non-finite or non-positive size parameter, or a refractive index whose real
/// part is not positive — inputs for which the recurrences have no meaning rather than poor
/// accuracy — and for a size parameter small enough that the upward recurrence for `χ`, which grows
/// as `(2n−1)!!/xⁿ`, overflows before it reaches `nₘₐₓ`. Bisected, that boundary sits at
/// `x = 1.0344774107715245e-60`; below it `ξₙ` is infinite and `aₙ` comes out `∞ − ∞`, and returning
/// those would give the caller a cross-section of `NaN` that no sign test downstream refuses.
fn coefficients(x: f64, m: Complex) -> Option<(Vec<Complex>, Vec<Complex>)> {
    if !x.is_finite() || x <= 0.0 || !m.re.is_finite() || m.re <= 0.0 || !m.im.is_finite() {
        return None;
    }
    let n_max = truncation_order(x);
    let y = m.scale(x);
    // Downward recurrence for D, started far enough above n_max that the seed is forgotten. The
    // margin is Wiscombe's (MIEV0) — a flat 25 plus `1.5√|y|` — rather than Bohren & Huffman's flat
    // 15, because the flat one does not hold at large size parameters: measured against `miepython`
    // 3.3.0, `+16` leaves `Q_ext` 5.7e-8 high at `x = 100` and 1.6e-4 high at `x = 1000`, where this
    // margin puts both inside 1.4e-13. The `|y|` term underneath is what makes either hold for an
    // absorbing sphere.
    let magnitude = libm::sqrt(y.norm_squared());
    let n_start = n_max.max(magnitude as usize) + 25 + (1.5 * libm::sqrt(magnitude)) as usize;
    let mut d = vec![Complex::new(0.0, 0.0); n_start + 1];
    for n in (1..=n_start).rev() {
        let n_over_y = y.inverse().scale(n as f64);
        d[n - 1] = n_over_y.sub(d[n].add(n_over_y).inverse());
    }

    let (mut psi_previous, mut psi) = (libm::cos(x), libm::sin(x));
    let (mut chi_previous, mut chi) = (-libm::sin(x), libm::cos(x));
    let mut a = Vec::with_capacity(n_max);
    let mut b = Vec::with_capacity(n_max);
    for (n, &d_n) in d.iter().enumerate().take(n_max + 1).skip(1) {
        let degree = n as f64;
        let psi_next = (2.0 * degree - 1.0) / x * psi - psi_previous;
        let chi_next = (2.0 * degree - 1.0) / x * chi - chi_previous;
        let (psi_n, psi_n_minus_1) = (psi_next, psi);
        let (chi_n, chi_n_minus_1) = (chi_next, chi);
        let xi_n = Complex::new(psi_n, -chi_n);
        let xi_n_minus_1 = Complex::new(psi_n_minus_1, -chi_n_minus_1);

        let n_over_x = Complex::new(degree / x, 0.0);
        let da = d_n.div(m).add(n_over_x);
        let db = m.mul(d_n).add(n_over_x);
        a.push(
            da.scale(psi_n)
                .sub(Complex::new(psi_n_minus_1, 0.0))
                .div(da.mul(xi_n).sub(xi_n_minus_1)),
        );
        b.push(
            db.scale(psi_n)
                .sub(Complex::new(psi_n_minus_1, 0.0))
                .div(db.mul(xi_n).sub(xi_n_minus_1)),
        );

        psi_previous = psi;
        psi = psi_next;
        chi_previous = chi;
        chi = chi_next;
    }

    // The recurrences are checked at their output, not at their input: `x` can be finite, positive
    // and still small enough that `χₙ` overflows, and a coefficient of `NaN` reported as success
    // reaches the caller as a cross-section of `NaN` that no downstream sign test refuses.
    if a.iter()
        .chain(b.iter())
        .any(|c| !c.re.is_finite() || !c.im.is_finite())
    {
        return None;
    }
    Some((a, b))
}

/// `(Q_ext, Q_sca, g·Q_sca)` from the coefficients of one sphere.
fn efficiencies(x: f64, a: &[Complex], b: &[Complex]) -> (f64, f64, f64) {
    let mut q_ext = 0.0;
    let mut q_sca = 0.0;
    let mut g_sum = 0.0;
    for (i, (an, bn)) in a.iter().zip(b.iter()).enumerate() {
        let n = (i + 1) as f64;
        q_ext += (2.0 * n + 1.0) * (an.re + bn.re);
        q_sca += (2.0 * n + 1.0) * (an.norm_squared() + bn.norm_squared());
        g_sum += (2.0 * n + 1.0) / (n * (n + 1.0)) * an.mul(bn.conjugate()).re;
        if i + 1 < a.len() {
            let (next_a, next_b) = (a[i + 1], b[i + 1]);
            g_sum += n * (n + 2.0) / (n + 1.0)
                * (an.mul(next_a.conjugate()).re + bn.mul(next_b.conjugate()).re);
        }
    }
    let scale = 2.0 / (x * x);
    (scale * q_ext, scale * q_sca, 4.0 / (x * x) * g_sum)
}

/// `(|S₁|² + |S₂|²)/2` at every node of `mu`, accumulated into `out` with weight `weight`.
fn accumulate_intensity(mu: &[f64], a: &[Complex], b: &[Complex], weight: f64, out: &mut [f64]) {
    for (node, slot) in mu.iter().zip(out.iter_mut()) {
        let x = *node;
        let (mut pi_previous, mut pi) = (0.0, 1.0);
        let mut s1 = Complex::new(0.0, 0.0);
        let mut s2 = Complex::new(0.0, 0.0);
        for (i, (an, bn)) in a.iter().zip(b.iter()).enumerate() {
            let n = (i + 1) as f64;
            let tau = n * x * pi - (n + 1.0) * pi_previous;
            let factor = (2.0 * n + 1.0) / (n * (n + 1.0));
            s1 = s1.add(an.scale(pi).add(bn.scale(tau)).scale(factor));
            s2 = s2.add(an.scale(tau).add(bn.scale(pi)).scale(factor));
            let next_pi = ((2.0 * n + 1.0) * x * pi - (n + 1.0) * pi_previous) / n;
            pi_previous = pi;
            pi = next_pi;
        }
        *slot += weight * 0.5 * (s1.norm_squared() + s2.norm_squared());
    }
}

/// Tabulate the size-integrated Mie phase function onto `mu`, returning the bulk optics.
///
/// `mu` is the cosine grid (ascending, `−1` to `+1`), `out` receives the *unnormalised* angular
/// intensity — [`TabulatedPhase::prepare`](crate::phase::TabulatedPhase::prepare) normalises it. The
/// refractive index is relative to the medium, in the `m = n + ik` convention with `k ≥ 0` for
/// absorption. `wavelength` sets the length unit shared with the distribution's radii.
///
/// Returns `None` if `mu` and `out` differ in length, if the wavelength is not positive, or if no
/// size in the quadrature yielded finite coefficients.
#[must_use]
pub fn phase_on_grid(
    distribution: &SizeDistribution,
    refractive_index: Complex,
    wavelength: f64,
    mu: &[f64],
    out: &mut [f64],
) -> Option<BulkOptics> {
    if mu.len() != out.len() || !wavelength.is_finite() || wavelength <= 0.0 {
        return None;
    }
    let (r_min, r_max) = distribution.radius_range();
    if r_min <= 0.0 || r_max <= r_min {
        return None;
    }
    for slot in out.iter_mut() {
        *slot = 0.0;
    }

    let k = core::f64::consts::TAU / wavelength;
    let log_step = (libm::log(r_max) - libm::log(r_min)) / (RADIUS_NODES - 1) as f64;
    let (mut extinction, mut scattering, mut g_scattering, mut number) = (0.0, 0.0, 0.0, 0.0);
    let mut any = false;

    for i in 0..RADIUS_NODES {
        // Simpson's rule in ln r; the extra r is the Jacobian of dr = r d(ln r).
        let simpson = if i == 0 || i == RADIUS_NODES - 1 {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let r = libm::exp(libm::log(r_min) + log_step * i as f64);
        let weight = simpson * log_step / 3.0 * distribution.density(r) * r;
        let x = k * r;
        let Some((a, b)) = coefficients(x, refractive_index) else {
            continue;
        };
        let (q_ext, q_sca, g_q_sca) = efficiencies(x, &a, &b);
        let geometric = core::f64::consts::PI * r * r;
        extinction += weight * q_ext * geometric;
        scattering += weight * q_sca * geometric;
        g_scattering += weight * g_q_sca * geometric;
        number += weight;
        // The angular intensity is (|S₁|²+|S₂|²)/2k², and the 1/k² is constant across sizes, so it
        // is left out of the accumulation the normalisation will divide away regardless.
        accumulate_intensity(mu, &a, &b, weight, out);
        any = true;
    }

    if !any || number <= 0.0 || scattering <= 0.0 {
        return None;
    }
    Some(BulkOptics {
        extinction_cross_section: extinction / number,
        scattering_cross_section: scattering / number,
        single_scattering_albedo: scattering / extinction,
        asymmetry: g_scattering / scattering,
    })
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::unreadable_literal,
        clippy::float_cmp,
        clippy::items_after_statements
    )]

    use super::*;

    /// The Mie series against `miepython`, an independent implementation.
    ///
    /// Recorded from `miepython` 3.3.0 (`efficiencies_mx`), which is a different code written from
    /// the same physics — the point of the check is that the recurrences here were transcribed
    /// correctly, and only an independent transcription can say so. `miepython` uses the `m = n − ik`
    /// sign convention, so its `1.33 − 0.01j` is this crate's `Complex::new(1.33, 0.01)`.
    ///
    /// The three cases above `x = 100` are the ones that matter, and they are here because an earlier
    /// version of this test stopped at `x = 100` and read its own 2e-8 residual as a difference of
    /// convention between the two codes. It was not: the downward recurrence for `Dₙ` was started too
    /// low, and at `x = 1000` that cost `Q_ext` 1.6e-4. A comparison that stops before the regime
    /// where a method degrades measures the regime, not the method.
    #[test]
    fn the_series_reproduces_an_independent_mie_implementation() {
        const CASES: [(f64, f64, f64, f64, f64, f64); 8] = [
            // x, n, k, Q_ext, Q_sca, g
            (
                0.1,
                1.33,
                0.0,
                1.1090625362212142e-5,
                1.1090625362212142e-5,
                1.8319588208688515e-3,
            ),
            (
                1.0,
                1.33,
                0.0,
                9.392400121407174e-2,
                9.392400121407174e-2,
                1.845166739820893e-1,
            ),
            (
                10.0,
                1.33,
                0.0,
                2.206548710184618,
                2.206548710184618,
                7.124592696732813e-1,
            ),
            (
                50.0,
                1.33,
                0.01,
                2.0888927642900703,
                1.239019566297607,
                9.391474988477505e-1,
            ),
            (
                100.0,
                1.5,
                0.1,
                2.089821842804492,
                1.1321339711247498,
                9.503916728871669e-1,
            ),
            (
                250.0,
                1.78,
                0.05,
                2.0496422278657196,
                1.1589200254813017,
                9.195310884848971e-1,
            ),
            (
                1000.0,
                1.33,
                0.0,
                2.0165783128481616,
                2.0165783128481616,
                8.830931644382337e-1,
            ),
            (
                2000.0,
                1.31,
                0.001,
                2.0124709903436098,
                1.0726583174985809,
                9.738991069081449e-1,
            ),
        ];
        // A few times the worst residual measured over these eight cases: `Q_ext` 1.6e-10, at
        // `x = 100` with `k = 0.1`, where it is a sum of nearly cancelling terms; `Q_sca` 1.4e-13;
        // and `g` 2.4e-11, at `x = 0.1`, where the Rayleigh regime cancels it the same way.
        const EXTINCTION_TOLERANCE: f64 = 1e-9;
        const SCATTERING_TOLERANCE: f64 = 1e-12;
        const ASYMMETRY_TOLERANCE: f64 = 1e-10;
        for (x, n, k, want_ext, want_sca, want_g) in CASES {
            let (a, b) = coefficients(x, Complex::new(n, k)).unwrap();
            let (q_ext, q_sca, g_q_sca) = efficiencies(x, &a, &b);
            assert!(
                (q_ext - want_ext).abs() < EXTINCTION_TOLERANCE * want_ext,
                "x={x}: Q_ext {q_ext} vs {want_ext}"
            );
            assert!(
                (q_sca - want_sca).abs() < SCATTERING_TOLERANCE * want_sca,
                "x={x}: Q_sca {q_sca} vs {want_sca}"
            );
            let g = g_q_sca / q_sca;
            assert!(
                (g - want_g).abs() < ASYMMETRY_TOLERANCE * want_g.abs(),
                "x={x}: g {g} vs {want_g}"
            );
        }
    }

    /// The Rayleigh limit, which is a closed form and not another code.
    ///
    /// As `x → 0` the series must collapse to `Q_sca = (8/3) x⁴ |(m²−1)/(m²+2)|²` and `g → 0`. That
    /// is the one regime where the answer is known analytically, and it exercises the whole chain —
    /// the logarithmic derivative, the Riccati–Bessel recurrences, the `a₁` coefficient — from a
    /// direction no reference implementation is involved in.
    #[test]
    fn the_rayleigh_limit_is_recovered() {
        for x in [1e-3, 1e-2] {
            let m = Complex::new(1.33, 0.0);
            let m2 = m.mul(m);
            let ratio = m2
                .sub(Complex::new(1.0, 0.0))
                .div(m2.add(Complex::new(2.0, 0.0)));
            let want = 8.0 / 3.0 * libm::pow(x, 4.0) * ratio.norm_squared();
            let (a, b) = coefficients(x, m).unwrap();
            let (_, q_sca, g_q_sca) = efficiencies(x, &a, &b);
            // The leading correction to the Rayleigh cross-section is O(x²), so the agreement is
            // expected to improve by a hundred between the two sizes rather than to be exact.
            assert!(
                (q_sca - want).abs() < 2.0 * x * x * want,
                "x={x}: Q_sca {q_sca} vs Rayleigh {want}"
            );
            assert!(
                (g_q_sca / q_sca).abs() < x,
                "x={x}: g = {}",
                g_q_sca / q_sca
            );
        }
    }

    /// The angular series and the efficiency series agree on how much light is scattered.
    ///
    /// `∫ (|S₁|² + |S₂|²)/2k² dΩ = C_sca` couples two code paths that share no arithmetic: the
    /// `π/τ` recursion with the amplitude sums on one side, the `Σ(2n+1)(|aₙ|²+|bₙ|²)` efficiency on
    /// the other. A transposed `S₁`/`S₂`, a wrong `(2n+1)/n(n+1)` weight or a `τₙ` recursion off by
    /// a term shows up here and in no reflectance.
    #[test]
    fn the_angular_series_carries_the_scattering_cross_section() {
        let x = 3.0;
        let m = Complex::new(1.4, 0.0);
        let (a, b) = coefficients(x, m).unwrap();
        let (_, q_sca, _) = efficiencies(x, &a, &b);

        const NODES: usize = 20_001;
        let mu: alloc::vec::Vec<f64> = (0..NODES)
            .map(|i| -1.0 + 2.0 * i as f64 / (NODES - 1) as f64)
            .collect();
        let mut intensity = alloc::vec![0.0; NODES];
        accumulate_intensity(&mu, &a, &b, 1.0, &mut intensity);

        // Simpson over mu; the solid-angle element is 2π dmu, and the 1/k² of the intensity is
        // written as 1/x² in units of the sphere radius so that both sides are efficiencies.
        let h = 2.0 / (NODES - 1) as f64;
        let mut integral = 0.0;
        for (i, value) in intensity.iter().enumerate() {
            let weight = if i == 0 || i == NODES - 1 {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            integral += weight * value;
        }
        integral *= h / 3.0 * core::f64::consts::TAU;
        let want = q_sca * core::f64::consts::PI * x * x;
        assert!(
            (integral - want).abs() < 1e-9 * want,
            "angular {integral} vs efficiency {want}"
        );
    }

    /// A cloud droplet distribution: the tabulated phase function's own asymmetry parameter
    /// reproduces the one the series computes in closed form, and the single-scattering albedo of
    /// non-absorbing water is 1.
    #[test]
    fn the_tabulated_asymmetry_reproduces_the_mie_series_value() {
        let distribution = SizeDistribution::Gamma {
            effective_radius: 6.0,
            effective_variance: 0.1,
        };
        const NODES: usize = 4001;
        let mu: alloc::vec::Vec<f64> = (0..NODES)
            .map(|i| {
                let theta = core::f64::consts::PI * (NODES - 1 - i) as f64 / (NODES - 1) as f64;
                libm::cos(theta).clamp(-1.0, 1.0)
            })
            .collect();
        let mut value = alloc::vec![0.0; NODES];
        let optics = phase_on_grid(
            &distribution,
            Complex::new(1.333, 0.0),
            0.55,
            &mu,
            &mut value,
        )
        .unwrap();
        assert!(
            (optics.single_scattering_albedo - 1.0).abs() < 1e-12,
            "non-absorbing water: omega = {}",
            optics.single_scattering_albedo
        );

        // Quadrature of the tabulated shape, against the closed-form series value.
        let (mut total, mut first) = (0.0, 0.0);
        for i in 0..NODES - 1 {
            let h = mu[i + 1] - mu[i];
            total += 0.5 * h * (value[i] + value[i + 1]);
            first += 0.5 * h * (mu[i] * value[i] + mu[i + 1] * value[i + 1]);
        }
        let tabulated = first / total;
        // The forward diffraction lobe of a 6 um droplet at 550 nm is about 5 degrees wide, so a
        // 0.045-degree grid resolves it; what remains is the quadrature error of that lobe.
        assert!(
            (tabulated - optics.asymmetry).abs() < 2e-3,
            "tabulated g {tabulated} vs series g {}",
            optics.asymmetry
        );
    }

    /// The forward peak a one-parameter fit cannot reach.
    ///
    /// Measured: a 6 um water droplet at 550 nm puts `p(0°) = 1365` where a Henyey–Greenstein of the
    /// *same* asymmetry parameter `g = 0.8528` puts `42.8` — a factor of **32**, in the direction
    /// that dominates a top-of-atmosphere radiance over cloud. That ratio is the whole reason this
    /// module exists, and it is a number rather than an assertion of principle.
    #[test]
    fn a_droplet_has_a_forward_peak_no_henyey_greenstein_of_the_same_g_can_match() {
        let distribution = SizeDistribution::Gamma {
            effective_radius: 6.0,
            effective_variance: 0.1,
        };
        const NODES: usize = 4001;
        let mu: alloc::vec::Vec<f64> = (0..NODES)
            .map(|i| {
                let theta = core::f64::consts::PI * (NODES - 1 - i) as f64 / (NODES - 1) as f64;
                libm::cos(theta).clamp(-1.0, 1.0)
            })
            .collect();
        let mut value = alloc::vec![0.0; NODES];
        let optics = phase_on_grid(
            &distribution,
            Complex::new(1.333, 0.0),
            0.55,
            &mu,
            &mut value,
        )
        .unwrap();

        let mut total = 0.0;
        for i in 0..NODES - 1 {
            total += 0.5 * (mu[i + 1] - mu[i]) * (value[i] + value[i + 1]);
        }
        let g = optics.asymmetry;
        let henyey = (1.0 - g * g) / (2.0 * libm::pow(1.0 + g * g - 2.0 * g, 1.5));
        let mie_forward = value[NODES - 1] / total;
        assert!(
            mie_forward > 25.0 * henyey,
            "forward peak {mie_forward} is not far above Henyey–Greenstein's {henyey} at g = {g}"
        );
    }

    /// Both distributions evaluate to their published forms, point by point.
    ///
    /// Every other use of `density` is inside a quadrature that normalises it away, so an exponent or
    /// a scale wrong by a factor survives every integral in this module. These are point values
    /// against the closed forms of Hansen & Travis (1974): the modified gamma
    /// `r^{(1−3v)/v} e^{−r/(rₑv)}` and the lognormal `e^{−ln²(r/r₀)/2σ²}/r`.
    #[test]
    fn both_size_distributions_evaluate_to_their_published_forms() {
        // v = 1/4 makes `(1 − 3v)/v` exactly 1 and `rₑv` exactly 5/2, so the reference below is exact
        // arithmetic and the tolerance measures the implementation rather than the constants.
        let gamma = SizeDistribution::Gamma {
            effective_radius: 10.0,
            effective_variance: 0.25,
        };
        for r in [0.5, 3.0, 17.0] {
            let want = r * libm::exp(-r / 2.5);
            assert!(
                (gamma.density(r) - want).abs() <= 1e-15 * want,
                "gamma at r = {r}: {} against {want}",
                gamma.density(r)
            );
        }
        let lognormal = SizeDistribution::Lognormal {
            median_radius: 2.0,
            log_sigma: 0.5,
        };
        for r in [0.4, 2.0, 9.0] {
            // Written as `ln²(r/r₀)/2σ²` rather than as the implementation's `t²/2`, so the two do
            // not share an arithmetic path.
            let logarithm = libm::log(r / 2.0);
            let want = libm::exp(-logarithm * logarithm / (2.0 * 0.25)) / r;
            assert!(
                (lognormal.density(r) - want).abs() <= 1e-14 * want,
                "lognormal at r = {r}: {} against {want}",
                lognormal.density(r)
            );
        }
    }

    /// The quadrature range is the one the doc comment states.
    ///
    /// The range decides which part of the distribution the optics see, and a range wrong by a factor
    /// still integrates to a plausible-looking cross-section — the tails it drops are exactly the
    /// ones no other check is sensitive to. Both ends of both members, against the arithmetic
    /// spelled out in `radius_range`'s own contract.
    #[test]
    fn the_quadrature_covers_the_stated_range_of_each_distribution() {
        let (lo, hi) = SizeDistribution::Gamma {
            effective_radius: 10.0,
            effective_variance: 0.25,
        }
        .radius_range();
        assert!((lo - 0.05).abs() <= 1e-15 * 0.05, "gamma lower {lo}");
        assert!((hi - 40.0).abs() <= 1e-15 * 40.0, "gamma upper {hi}");

        let (lo, hi) = SizeDistribution::Lognormal {
            median_radius: 2.0,
            log_sigma: 0.5,
        }
        .radius_range();
        let (want_lo, want_hi) = (2.0 * libm::exp(-2.5), 2.0 * libm::exp(2.5));
        assert!(
            (lo - want_lo).abs() <= 1e-15 * want_lo,
            "lognormal lower {lo}"
        );
        assert!(
            (hi - want_hi).abs() <= 1e-15 * want_hi,
            "lognormal upper {hi}"
        );
    }

    /// The series is cut where Wiscombe says, and carries exactly that many terms.
    ///
    /// The truncation order is a published criterion, and the terms past it are small enough that
    /// dropping one — or keeping thirty more — moves no cross-section by as much as a comparison
    /// against `miepython` can see. What the order *is* therefore has to be asserted directly, and
    /// the length of the coefficient vectors asserted against it, or the two can drift apart
    /// silently.
    #[test]
    fn the_series_is_truncated_at_wiscombes_order() {
        // ⌊x + 4x^{1/3} + 2⌋ + 1, Wiscombe (1980) — evaluated by hand at each x.
        for (x, want) in [
            (0.1, 4usize),
            (1.0, 8),
            (10.0, 21),
            (50.0, 67),
            (100.0, 121),
        ] {
            assert_eq!(truncation_order(x), want, "order at x = {x}");
            let (a, b) = coefficients(x, Complex::new(1.33, 0.01)).unwrap();
            assert_eq!(a.len(), want, "a terms at x = {x}");
            assert_eq!(b.len(), want, "b terms at x = {x}");
        }
    }

    /// The composite Simpson rule, reassembled panel by panel.
    ///
    /// `phase_on_grid` walks the radius grid once and picks each node's weight from the
    /// `1-4-2-…-4-1` pattern; this builds the same rule from its definition instead, adding
    /// `(h/3)(f₀ + 4f₁ + f₂)` over each pair of intervals and letting the shared nodes accumulate.
    /// Same quadrature, assembled the other way round, so a weight on the wrong node, a node at the
    /// wrong radius, or a step over the wrong number of intervals separates the two.
    ///
    /// The Mie half is shared deliberately — `coefficients` and `efficiencies` are already sealed
    /// against `miepython`, and repeating them here would test nothing. What is *not* shared is the
    /// scale: `out` is documented as unnormalised, so its absolute size is not a physical quantity,
    /// but it is the accumulated quadrature weight, and comparing it is the only thing that makes a
    /// mutation which merely rescales every node visible at all.
    fn panelwise_reference(
        distribution: &SizeDistribution,
        m: Complex,
        wavelength: f64,
        mu: &[f64],
        out: &mut [f64],
    ) -> BulkOptics {
        let (r_min, r_max) = distribution.radius_range();
        let step = libm::log(r_max / r_min) / (RADIUS_NODES - 1) as f64;
        let mut quadrature = [0.0; RADIUS_NODES];
        for panel in 0..(RADIUS_NODES - 1) / 2 {
            quadrature[2 * panel] += step / 3.0;
            quadrature[2 * panel + 1] += 4.0 * step / 3.0;
            quadrature[2 * panel + 2] += step / 3.0;
        }

        let k = core::f64::consts::TAU / wavelength;
        let (mut extinction, mut scattering, mut g_scattering, mut number) = (0.0, 0.0, 0.0, 0.0);
        for (i, &node_weight) in quadrature.iter().enumerate() {
            let r = r_min * libm::exp(step * i as f64);
            let weight = node_weight * distribution.density(r) * r;
            let x = k * r;
            let (a, b) = coefficients(x, m).unwrap();
            let (q_ext, q_sca, g_q_sca) = efficiencies(x, &a, &b);
            let area = core::f64::consts::PI * r * r;
            extinction += weight * q_ext * area;
            scattering += weight * q_sca * area;
            g_scattering += weight * g_q_sca * area;
            number += weight;
            accumulate_intensity(mu, &a, &b, weight, out);
        }
        BulkOptics {
            extinction_cross_section: extinction / number,
            scattering_cross_section: scattering / number,
            single_scattering_albedo: scattering / extinction,
            asymmetry: g_scattering / scattering,
        }
    }

    /// The size integration is the rule it says it is, on both members.
    #[test]
    fn the_size_integration_is_the_composite_simpson_rule_it_claims_to_be() {
        const NODES: usize = 91;
        let mu: alloc::vec::Vec<f64> = (0..NODES)
            .map(|i| {
                let theta = core::f64::consts::PI * (NODES - 1 - i) as f64 / (NODES - 1) as f64;
                libm::cos(theta).clamp(-1.0, 1.0)
            })
            .collect();
        let cases = [
            (
                SizeDistribution::Gamma {
                    effective_radius: 6.0,
                    effective_variance: 0.1,
                },
                Complex::new(1.333, 0.0),
            ),
            (
                SizeDistribution::Lognormal {
                    median_radius: 0.3,
                    log_sigma: 0.7,
                },
                Complex::new(1.53, 0.008),
            ),
        ];
        for (distribution, m) in cases {
            let mut got_out = alloc::vec![0.0; NODES];
            let got = phase_on_grid(&distribution, m, 0.55, &mu, &mut got_out).unwrap();
            let mut want_out = alloc::vec![0.0; NODES];
            let want = panelwise_reference(&distribution, m, 0.55, &mu, &mut want_out);
            for (name, a, b) in [
                (
                    "C_ext",
                    got.extinction_cross_section,
                    want.extinction_cross_section,
                ),
                (
                    "C_sca",
                    got.scattering_cross_section,
                    want.scattering_cross_section,
                ),
                (
                    "omega",
                    got.single_scattering_albedo,
                    want.single_scattering_albedo,
                ),
                ("g", got.asymmetry, want.asymmetry),
            ] {
                assert!(
                    (a - b).abs() <= 1e-10 * b.abs(),
                    "{name}: {a} against the panelwise {b}"
                );
            }
            for (i, (&a, &b)) in got_out.iter().zip(want_out.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-10 * b.abs(),
                    "node {i}: {a} against the panelwise {b}"
                );
            }
        }
    }

    /// A distribution that scatters nothing is refused rather than returned as a `0/0`.
    ///
    /// The leading Mie coefficient goes as `x³` and the cross-sections as `|aₙ|²`, so the two ways an
    /// arbitrarily small particle breaks the arithmetic sit six decades apart in size parameter and
    /// are caught by two different guards. Both end in `None`, and the number density — which carries
    /// no power of `x` — stays positive throughout, so neither is caught by accident.
    ///
    /// * `rₑ = 1e-57` puts `x` between `7e-59` and `5e-56`. Every coefficient is finite, but `|a₁|²`
    ///   is around `1e-360` and underflows to exactly zero, so the scattering cross-section is zero
    ///   while the extinction cross-section is not: the albedo and the asymmetry parameter would both
    ///   be `0/0`.
    /// * `rₑ = 1e-99` is three decades below the recurrence's own floor, so no size in the quadrature
    ///   yields coefficients at all and nothing is accumulated.
    #[test]
    fn a_distribution_whose_scattering_underflows_is_refused() {
        for effective_radius in [1e-57, 1e-99] {
            let distribution = SizeDistribution::Gamma {
                effective_radius,
                effective_variance: 0.3,
            };
            let mut out = [0.0; 3];
            assert!(
                phase_on_grid(
                    &distribution,
                    Complex::new(1.33, 0.0),
                    0.55,
                    &[-1.0, 0.0, 1.0],
                    &mut out
                )
                .is_none(),
                "rₑ = {effective_radius:e}"
            );
        }
    }

    /// The inputs the series has no meaning for are refused rather than returned as numbers.
    #[test]
    fn unphysical_inputs_are_refused() {
        // Each disjunct of `coefficients`' guard, asked with an argument the *output* check cannot
        // stand in for. That distinction is the whole of this list: since #287 the function also
        // refuses a coefficient that came out non-finite, so every non-finite argument would be
        // rejected with the input guard gone — but a finite negative one would not. `x = -1` and
        // `m.re = -1.33` run the recurrences to completion and return four plausible, finite, wrong
        // coefficients (`a₁ = 0.0153 + 0.1228i` at `x = -1`), and `m.re = +∞` panics
        // `truncation_order` with an integer overflow. The guard is the only thing between a caller
        // and either.
        for x in [
            -1.0,
            -1e-3,
            -10.0,
            0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert!(
                coefficients(x, Complex::new(1.33, 0.0)).is_none(),
                "x = {x}"
            );
        }
        for re in [-1.33, -1.0, 0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(
                coefficients(1.0, Complex::new(re, 0.0)).is_none(),
                "m.re = {re}"
            );
        }
        for im in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(
                coefficients(1.0, Complex::new(1.33, im)).is_none(),
                "m.im = {im}"
            );
        }
        let mut out = [0.0; 4];
        let distribution = SizeDistribution::Lognormal {
            median_radius: 0.1,
            log_sigma: 0.7,
        };
        let m = Complex::new(1.45, 0.005);
        assert!(phase_on_grid(&distribution, m, 0.55, &[-1.0, 1.0], &mut out).is_none());
        assert!(phase_on_grid(&distribution, m, 0.0, &[-1.0, 0.0, 0.5, 1.0], &mut out).is_none());
    }

    /// Absorbing aerosol has an albedo strictly inside `(0, 1)` and a forward-peaked phase function,
    /// so the lognormal branch and the absorbing branch are both exercised on a realistic case.
    #[test]
    fn absorbing_aerosol_has_an_albedo_between_zero_and_one() {
        let distribution = SizeDistribution::Lognormal {
            median_radius: 0.3,
            log_sigma: 0.7,
        };
        let mu: alloc::vec::Vec<f64> = (0..361)
            .map(|i| {
                let theta = core::f64::consts::PI * (360 - i) as f64 / 360.0;
                libm::cos(theta).clamp(-1.0, 1.0)
            })
            .collect();
        let mut value = alloc::vec![0.0; mu.len()];
        let optics = phase_on_grid(
            &distribution,
            Complex::new(1.53, 0.008),
            0.55,
            &mu,
            &mut value,
        )
        .unwrap();
        assert!(
            (0.7..0.99).contains(&optics.single_scattering_albedo),
            "omega = {}",
            optics.single_scattering_albedo
        );
        assert!(optics.asymmetry > 0.5, "g = {}", optics.asymmetry);
        assert!(optics.scattering_cross_section < optics.extinction_cross_section);
    }
}
