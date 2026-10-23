//! Atmospheric correction (TOA → surface reflectance) and a kernel-driven surface BRDF.
//!
//! A sensor measures top-of-atmosphere reflectance `ρ_TOA`; the surface reflectance `ρ_s` is what
//! remote sensing wants. The 6S / MODTRAN coupling relates them through four atmosphere terms — path
//! reflectance `ρ_atm`, downward and upward transmittances `T↓`, `T↑`, and spherical albedo `s`:
//!
//! ```text
//! ρ_TOA = ρ_atm + T↓·T↑·ρ_s / (1 − s·ρ_s).
//! ```
//!
//! [`AtmosphericCoupling`] holds those terms and inverts the relation *exactly*
//! ([`AtmosphericCoupling::surface_from_toa`]); [`coupling_from_slab`] derives the terms from the
//! Monte-Carlo slab radiative transfer of [`crate::radiative_transfer`]: the path reflectance from the
//! directional (next-event) estimator at the full sun/view/azimuth geometry, the transmittances from
//! analog runs at the sun and view cosines, the spherical albedo by a flux-weighted hemispheric integral.
//!
//! `ρ_atm` is bidirectional — it is what the sensor sees over a black surface, along one line of sight —
//! while `T↓`, `T↑` and `s` are flux quantities and azimuthally symmetric in a plane-parallel layer.
//! Substituting the hemispheric plane albedo for `ρ_atm` is an error of the order of the phase
//! function's anisotropy, and it does not average out over a scene: measured on a Rayleigh layer at
//! `τ = 0.1`, `ω = 1`, `μ_s = 0.8`, `μ_v = 0.6`, the plane albedo runs **39.9 % high** in the
//! forward-scattering half-plane and **20.0 % low** in the backscattering one — which retrieves 0.081
//! and 0.117 for a surface whose reflectance is 0.100, a ±20 % error systematic with scan angle.
//!
//! The surface is not Lambertian: its reflectance depends on the sun/view geometry through its
//! bidirectional reflectance distribution function. [`brdf_reflectance`] is the MODIS MCD43
//! semi-empirical **RTLSR** model `ρ = f_iso + f_vol·K_vol + f_geo·K_geo` with the RossThick volumetric
//! ([`ross_thick_kernel`]) and LiSparse-Reciprocal geometric ([`li_sparse_reciprocal_kernel`]) kernels.

// ρ/TOA/BOA/RossThick/LiSparse/6S/MODIS are domain terms; short angle names mirror the BRDF literature.
#![allow(
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::unreadable_literal
)]

use alloc::vec::Vec;

use crate::radiative_transfer::{Atmosphere, Layer, monte_carlo_stack, stack_path_reflectance};

/// The four 6S atmosphere terms coupling surface and top-of-atmosphere reflectance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtmosphericCoupling {
    /// Atmospheric path reflectance `ρ_atm` (the TOA reflectance over a black surface).
    pub path_reflectance: f64,
    /// Downward total (direct + diffuse) transmittance sun → surface, `T↓`.
    pub transmittance_down: f64,
    /// Upward total transmittance surface → sensor, `T↑`.
    pub transmittance_up: f64,
    /// Atmospheric spherical albedo `s` (the surface–atmosphere multiple-reflection factor).
    pub spherical_albedo: f64,
}

impl AtmosphericCoupling {
    /// Forward: the TOA reflectance a surface of reflectance `surface_reflectance` produces.
    #[must_use]
    pub fn toa_from_surface(&self, surface_reflectance: f64) -> f64 {
        self.path_reflectance
            + self.transmittance_down * self.transmittance_up * surface_reflectance
                / (1.0 - self.spherical_albedo * surface_reflectance)
    }

    /// Inverse (atmospheric correction): the surface reflectance behind a measured `toa_reflectance`.
    /// Exact inverse of [`Self::toa_from_surface`]: with `y = (ρ_TOA − ρ_atm)/(T↓T↑)`,
    /// `ρ_s = y/(1 + s·y)`.
    #[must_use]
    pub fn surface_from_toa(&self, toa_reflectance: f64) -> f64 {
        let y = (toa_reflectance - self.path_reflectance)
            / (self.transmittance_down * self.transmittance_up);
        y / (1.0 + self.spherical_albedo * y)
    }
}

/// Four-node Gauss–Legendre quadrature on `[0, 1]` — `(node, weight)` — for the spherical-albedo flux
/// integral `∫₀¹ R(μ) 2μ dμ`.
const GAUSS_01: [(f64, f64); 4] = [
    (0.06943184420297367, 0.1739274225687269),
    (0.33000947820757185, 0.32607257743127305),
    (0.6699905217924281, 0.32607257743127305),
    (0.9305681557970263, 0.1739274225687269),
];

/// Derive the [`AtmosphericCoupling`] from the Monte-Carlo slab radiative transfer: path reflectance
/// from the directional estimator at the full `(mu_sun, mu_view, relative_azimuth)` geometry, downward
/// transmittance from a run at the solar cosine `mu_sun`, upward transmittance from a run at the view
/// cosine `mu_view`, and spherical albedo from the flux-weighted hemispheric reflectance
/// `∫₀¹ R(μ) 2μ dμ`. `photons` and `seed` control the (deterministic) Monte-Carlo streams.
///
/// `relative_azimuth` follows `path_reflectance`: measured from the forward-scattering half-plane,
/// so `π` with `mu_sun == mu_view` is exact backscatter. Only `ρ_atm` reads it — the two transmittances
/// and the spherical albedo are azimuthally symmetric in a plane-parallel layer, which is why they are
/// flux integrals of an analog run and `ρ_atm` cannot be.
#[must_use]
pub fn coupling_from_slab(
    atmosphere: &Atmosphere,
    mu_sun: f64,
    mu_view: f64,
    relative_azimuth: f64,
    photons: u32,
    seed: u64,
) -> AtmosphericCoupling {
    coupling_from_stack(
        &[atmosphere.layer()],
        mu_sun,
        mu_view,
        relative_azimuth,
        photons,
        seed,
    )
}

/// [`coupling_from_slab`] over a vertically stratified stack, ordered top down — the same six runs
/// at the same seed derivations, through [`monte_carlo_stack`] and [`stack_path_reflectance`].
///
/// This is where a real atmosphere differs from the one [`Atmosphere`] can express: Rayleigh
/// scattering follows the air, aerosol sits in the boundary layer, and mixing them into one
/// effective medium puts molecular scattering too low and particulate scattering too high. What that
/// is worth is measured, not assumed — `examples/layer_arrangement.rs` differences the two
/// arrangements at a shared seed over RadCalNet's own scenes and reports the spread.
///
/// A one-slice stack is [`coupling_from_slab`], bit for bit; that identity is what the shipped
/// single-layer answer rides on, and it is asserted rather than argued.
#[must_use]
pub fn coupling_from_stack(
    layers: &[Layer],
    mu_sun: f64,
    mu_view: f64,
    relative_azimuth: f64,
    photons: u32,
    seed: u64,
) -> AtmosphericCoupling {
    let sun = monte_carlo_stack(layers, mu_sun, photons, seed);
    let view = monte_carlo_stack(layers, mu_view, photons, seed ^ 0x9e37_79b9);
    let mut spherical = 0.0;
    for (i, (mu, w)) in GAUSS_01.iter().enumerate() {
        let r =
            monte_carlo_stack(layers, *mu, photons, seed.wrapping_add(101 + i as u64)).reflectance;
        spherical += w * r * 2.0 * mu;
    }
    AtmosphericCoupling {
        path_reflectance: stack_path_reflectance(
            layers,
            mu_sun,
            mu_view,
            relative_azimuth,
            photons,
            seed.wrapping_add(3),
        ),
        transmittance_down: sun.transmittance,
        transmittance_up: view.transmittance,
        spherical_albedo: spherical,
    }
}

/// The RossThick volumetric BRDF kernel at solar zenith `theta_s`, view zenith `theta_v` and relative
/// azimuth `rel_azimuth` (all radians): `K_vol = [(π/2 − ξ)cos ξ + sin ξ]/(cos θs + cos θv) − π/4`, with
/// the phase angle `ξ` from `cos ξ = cos θs cos θv + sin θs sin θv cos φ`.
#[must_use]
pub fn ross_thick_kernel(theta_s: f64, theta_v: f64, rel_azimuth: f64) -> f64 {
    let (cs, ss) = (libm::cos(theta_s), libm::sin(theta_s));
    let (cv, sv) = (libm::cos(theta_v), libm::sin(theta_v));
    let cos_xi = (cs * cv + ss * sv * libm::cos(rel_azimuth)).clamp(-1.0, 1.0);
    let xi = libm::acos(cos_xi);
    ((core::f64::consts::FRAC_PI_2 - xi) * cos_xi + libm::sin(xi)) / (cs + cv)
        - core::f64::consts::FRAC_PI_4
}

/// The LiSparse-Reciprocal geometric BRDF kernel (radians), with the MODIS crown-shape parameters
/// `h/b = 2`, `b/r = 1`. Models mutual shadowing of protrusions on the surface.
#[must_use]
pub fn li_sparse_reciprocal_kernel(theta_s: f64, theta_v: f64, rel_azimuth: f64) -> f64 {
    const HB: f64 = 2.0;
    const BR: f64 = 1.0;
    let ts = libm::atan(BR * libm::tan(theta_s));
    let tv = libm::atan(BR * libm::tan(theta_v));
    let (cs, ss, tts) = (libm::cos(ts), libm::sin(ts), libm::tan(ts));
    let (cv, sv, ttv) = (libm::cos(tv), libm::sin(tv), libm::tan(tv));
    let (cphi, sphi) = (libm::cos(rel_azimuth), libm::sin(rel_azimuth));
    let cos_xi = cs * cv + ss * sv * cphi;
    let d2 = tts * tts + ttv * ttv - 2.0 * tts * ttv * cphi;
    let sec = 1.0 / cs + 1.0 / cv;
    let cos_t =
        (HB * libm::sqrt(d2 + (tts * ttv * sphi) * (tts * ttv * sphi)) / sec).clamp(-1.0, 1.0);
    let t = libm::acos(cos_t);
    let overlap = (t - libm::sin(t) * cos_t) * sec / core::f64::consts::PI;
    overlap - sec + 0.5 * (1.0 + cos_xi) / cs / cv
}

/// The MODIS RTLSR surface reflectance `ρ = f_iso + f_vol·K_vol + f_geo·K_geo` at the given geometry
/// (radians), from the isotropic/volumetric/geometric kernel weights.
#[must_use]
pub fn brdf_reflectance(
    f_iso: f64,
    f_vol: f64,
    f_geo: f64,
    theta_s: f64,
    theta_v: f64,
    rel_azimuth: f64,
) -> f64 {
    f_iso
        + f_vol * ross_thick_kernel(theta_s, theta_v, rel_azimuth)
        + f_geo * li_sparse_reciprocal_kernel(theta_s, theta_v, rel_azimuth)
}

/// How many halvings [`CorrelatedKCoupling::surface_from_toa`] spends. The bracket is 2 wide, so 60
/// halvings take it to 2e-18 — below the last bit of any reflectance in `[−1, 1]`, and a fixed count
/// rather than a tolerance because a monotone bisection on a bracket of known width has no other
/// termination question to answer.
const BISECTIONS: u32 = 60;

/// The 6S coupling of a channel whose gas absorbs in bands: one [`AtmosphericCoupling`] per term of
/// the channel's k-distribution, with the term's weight.
///
/// # Why not one averaged coupling
///
/// Averaging the four terms and coupling once would be wrong, and wrong in exactly the way the
/// k-distribution exists to avoid. `ρ_TOA` is not linear in the coupling: the surface–atmosphere
/// factor `1/(1 − s ρ_s)` and, more importantly, the whole point that a band's transmittance is an
/// *average of exponentials* mean that
///
/// ```text
///     sum_j w_j f(coupling_j)  !=  f(sum_j w_j coupling_j)
/// ```
///
/// with a gap of the order of the spread in `T↓T↑` across the terms — which in a band core is the
/// difference between opaque and transparent. So the terms are carried and the *observable* is
/// weight-summed, which is the composition the k-distribution is a decomposition for.
#[derive(Clone, Debug)]
pub struct CorrelatedKCoupling {
    terms: Vec<(f64, AtmosphericCoupling)>,
}

impl CorrelatedKCoupling {
    /// Forward: the TOA reflectance a surface of reflectance `surface_reflectance` produces, summed
    /// over the k-distribution.
    #[must_use]
    pub fn toa_from_surface(&self, surface_reflectance: f64) -> f64 {
        self.terms
            .iter()
            .map(|(weight, coupling)| weight * coupling.toa_from_surface(surface_reflectance))
            .sum()
    }

    /// Inverse (atmospheric correction): the surface reflectance behind a measured `toa_reflectance`.
    ///
    /// A sum of Möbius functions with different poles is a rational function of the term count, so
    /// unlike the single-term [`AtmosphericCoupling::surface_from_toa`] there is no closed form. It is
    /// solved instead, by bisection over `ρ_s ∈ [−1, 1]`: [`Self::toa_from_surface`] is strictly
    /// increasing there for any physical coupling, so the root is unique and bracketed.
    ///
    /// The bracket runs below zero deliberately. A negative retrieval is what a TOA reflectance darker
    /// than the path reflectance *means*, and clamping it to zero would turn a scene whose inputs are
    /// wrong into a scene that merely looks dark — the same reason
    /// [`AtmosphericCoupling::surface_from_toa`] inverts exactly rather than clamping.
    ///
    /// Returns `None` when `toa_reflectance` lies outside `[toa_from_surface(−1),
    /// toa_from_surface(1)]`, which no surface can produce under this coupling.
    #[must_use]
    pub fn surface_from_toa(&self, toa_reflectance: f64) -> Option<f64> {
        let (mut lo, mut hi) = (-1.0_f64, 1.0_f64);
        if !(self.toa_from_surface(lo)..=self.toa_from_surface(hi)).contains(&toa_reflectance) {
            return None;
        }
        for _ in 0..BISECTIONS {
            let mid = f64::midpoint(lo, hi);
            if self.toa_from_surface(mid) <= toa_reflectance {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(f64::midpoint(lo, hi))
    }

    /// The weights and couplings, in the order [`correlated_k_coupling`] was handed them.
    #[must_use]
    pub fn terms(&self) -> &[(f64, AtmosphericCoupling)] {
        &self.terms
    }
}

/// Photons for one term of a `terms`-term k-distribution, out of a budget of `photons` per term on
/// average, divided in proportion to `weight`.
///
/// The weighted sum `Σ wⱼ Xⱼ` has variance `Σ wⱼ² σ²/nⱼ`, and under a fixed `Σ nⱼ` that is smallest
/// at `nⱼ ∝ wⱼ`. So the equal split is not a neutral default but the *worst* allocation the same
/// budget can buy, and the factor between the two is `m Σ wⱼ²` — at least 1 by Cauchy–Schwarz, and
/// equal to 1 only where the weights are already uniform. **No channel can come out noisier than the
/// equal split**, which is what makes this safe to apply everywhere rather than behind a threshold.
///
/// Measured over the shipped 1055-row gas table at 256 bins, the variance falls 3.54× at the median
/// channel and 8.10× in the mean, reaching 90.7× at 1900 nm — a strong band puts nearly all of its
/// weight on a handful of depths — and 1.07× at 2350 nm, where the weights are near-uniform and
/// there is nothing to redistribute.
///
/// The total is unchanged, so this buys accuracy at the same cost. A caller who wants the saving as
/// speed instead passes a smaller `photons`; the parameter's units do not change, so that trade
/// stays the caller's to make.
///
/// The floor of one photon keeps a negligible term negligible rather than absent: dropping it would
/// break `Σ w = 1`, and on the worst channel of the shipped table the terms the floor catches carry
/// 2.8e-4 of that channel's weight, against the 3.9e-3 the resort itself costs.
fn term_photons(photons: u32, terms: usize, weight: f64) -> u32 {
    #[expect(clippy::cast_precision_loss, reason = "term counts are far below 2^53")]
    let share = libm::round(f64::from(photons) * terms as f64 * weight);
    // Written as two saturating steps rather than an `if share < 1.0` because the branch form has a
    // tie at exactly one photon that no test can observe — both arms return 1 — so it ships an
    // equivalent mutant and an equivalence claim to defend. `f64::max` returns the other operand
    // when one is NaN, which lands a weight the caller never validated on the floor instead of a
    // panic or a zero; returning zero would drop the term's draw while leaving its weight in the sum.
    let bounded = share.max(1.0).min(f64::from(u32::MAX));
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "bounded into [1, u32::MAX] on the line above"
    )]
    let allocated = bounded as u32;
    allocated
}

/// Derive the [`CorrelatedKCoupling`] of a scattering slab plus a channel's gaseous absorption: run
/// [`coupling_from_slab`] once per k-distribution term, on the slab that term's grey absorber makes
/// ([`Atmosphere::with_absorber`]).
///
/// `gas_optical_depths` is `(weight, vertical gaseous optical depth)` from
/// [`crate::gas::gas_optical_depths`]. An empty slice is a channel with no absorber and gives the
/// bare `atmosphere` back as a single unit-weight term, so a caller need not special-case a window.
///
/// Each term gets its own Monte-Carlo stream, `seed` offset by the term index. Sharing one stream
/// across the terms would correlate their errors, and correlated errors do not average down in the
/// weighted sum — they are what a k-distribution's variance looks like when the terms are not
/// independent draws.
///
/// `photons` is the budget **per term on average**, not per term: the total is `photons` times the
/// term count either way, and `term_photons` decides how it is divided.
#[must_use]
pub fn correlated_k_coupling(
    atmosphere: &Atmosphere,
    gas_optical_depths: &[(f64, f64)],
    mu_sun: f64,
    mu_view: f64,
    relative_azimuth: f64,
    photons: u32,
    seed: u64,
) -> CorrelatedKCoupling {
    let bare = [(1.0, 0.0)];
    let terms = if gas_optical_depths.is_empty() {
        &bare[..]
    } else {
        gas_optical_depths
    };
    CorrelatedKCoupling {
        terms: terms
            .iter()
            .enumerate()
            .map(|(index, &(weight, depth))| {
                let layer = atmosphere.with_absorber(depth);
                let coupling = coupling_from_slab(
                    &layer,
                    mu_sun,
                    mu_view,
                    relative_azimuth,
                    term_photons(photons, terms.len(), weight),
                    seed.wrapping_add((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                );
                (weight, coupling)
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unreadable_literal,
        clippy::float_cmp,
        clippy::unwrap_used,
        clippy::expect_used
    )]
    use super::*;
    use crate::radiative_transfer::monte_carlo_slab;

    /// The whole coupling, from a uniform medium cut twelve ways, must be the undivided one field
    /// for field.
    ///
    /// [`coupling_from_slab`] delegates, so the one-slice case is an identity by construction and
    /// worth nothing as a test; twelve slices is the claim, and it holds across a `tau` that differs
    /// by one ulp — see `a_uniform_stack_reproduces_the_undivided_layer`. All six runs are involved —
    /// the two transmittances, the four quadrature nodes of the spherical albedo, and the
    /// next-event path reflectance — so this is what says the stratified path can be substituted
    /// for the shipped one wherever the medium happens to be uniform, and that every number already
    /// published under `coupling_from_slab` survives the refactor unchanged.
    #[test]
    fn a_uniform_stack_reproduces_the_undivided_coupling() {
        let atm = Atmosphere {
            optical_depth: 0.24,
            single_scattering_albedo: 0.97,
            phase: PhaseFunction::Rayleigh,
        };
        let cut: [Layer; 12] = core::array::from_fn(|_| Layer {
            thickness: atm.optical_depth / 12.0,
            single_scattering_albedo: atm.single_scattering_albedo,
            phase: atm.phase,
        });
        let (mu_sun, mu_view, photons, seed) = (0.82, 0.64, 20_000u32, 20_260_821u64);

        assert_eq!(
            coupling_from_slab(&atm, mu_sun, mu_view, 0.3, photons, seed),
            coupling_from_stack(&cut, mu_sun, mu_view, 0.3, photons, seed),
            "twelve uniform slices must reproduce the whole coupling"
        );
    }
    use crate::phase::PhaseFunction;

    /// The path reflectance is bidirectional, and the plane albedo is not a substitute for it.
    ///
    /// This is the whole content of the term: `ρ_atm` is the one place the 6S coupling reads the view
    /// azimuth, so a run at `Δφ = 0` and one at `Δφ = π` must differ there and be **bit-identical**
    /// everywhere else — `T↓`, `T↑` and `s` are flux integrals over a plane-parallel layer and cannot
    /// depend on an azimuth. Rayleigh peaks at backscatter, so `Δφ = π` is the brighter one.
    ///
    /// Then the cost of the substitution, priced where it is paid: correcting a `ρ_s = 0.1` surface
    /// with the plane albedo in place of the true `ρ_atm` retrieves 0.081 in the forward half-plane and
    /// 0.117 in the backward one. That is a ±20 % error on the quantity the module exists to produce,
    /// systematic with scan angle, and it survives any amount of averaging over the scene.
    #[test]
    fn the_path_reflectance_is_the_one_term_that_reads_the_view_azimuth() {
        let atm = Atmosphere {
            optical_depth: 0.1,
            single_scattering_albedo: 1.0,
            phase: PhaseFunction::Rayleigh,
        };
        let (mu_sun, mu_view, photons, seed) = (0.8, 0.6, 300_000u32, 2026u64);
        let forward = coupling_from_slab(&atm, mu_sun, mu_view, 0.0, photons, seed);
        let backward =
            coupling_from_slab(&atm, mu_sun, mu_view, core::f64::consts::PI, photons, seed);

        assert!(
            backward.path_reflectance > forward.path_reflectance * 1.5,
            "Rayleigh backscatters: {} against {}",
            backward.path_reflectance,
            forward.path_reflectance
        );
        assert_eq!(forward.transmittance_down, backward.transmittance_down);
        assert_eq!(forward.transmittance_up, backward.transmittance_up);
        assert_eq!(forward.spherical_albedo, backward.spherical_albedo);

        // The plane albedo the term used to carry: the hemispheric reflectance of the same layer.
        let plane = monte_carlo_slab(&atm, mu_sun, photons, seed).reflectance;
        let wrong = |truth: &AtmosphericCoupling| {
            let mistaken = AtmosphericCoupling {
                path_reflectance: plane,
                ..*truth
            };
            mistaken.surface_from_toa(truth.toa_from_surface(0.1))
        };
        let (low, high) = (wrong(&forward), wrong(&backward));
        assert!(
            (low - 0.081).abs() < 0.002 && (high - 0.117).abs() < 0.002,
            "a 0.1 surface retrieves {low} forward and {high} backward"
        );
    }

    #[test]
    fn six_s_inversion_exactly_round_trips() {
        let atm = AtmosphericCoupling {
            path_reflectance: 0.08,
            transmittance_down: 0.85,
            transmittance_up: 0.88,
            spherical_albedo: 0.12,
        };
        for rho_s in [0.02, 0.05, 0.2, 0.5, 0.8] {
            let toa = atm.toa_from_surface(rho_s);
            let recovered = atm.surface_from_toa(toa);
            assert!((recovered - rho_s).abs() < 1e-15, "{recovered} vs {rho_s}");
        }
        // A dark surface: path radiance dominates, so the atmosphere brightens the TOA signal, and the
        // correction must pull it back down.
        assert!(atm.toa_from_surface(0.02) > 0.02);
    }

    #[test]
    fn coupling_from_a_thin_atmosphere_is_near_transparent() {
        // τ → 0: negligible path reflectance, ~unit transmittance, ~zero spherical albedo, so the
        // retrieved surface reflectance is essentially the TOA value.
        let thin = Atmosphere {
            optical_depth: 1e-4,
            single_scattering_albedo: 0.9,
            phase: PhaseFunction::HenyeyGreenstein { asymmetry: 0.0 },
        };
        let c = coupling_from_slab(&thin, 0.8, 0.9, 1.5, 40000, 12345);
        assert!(c.path_reflectance < 5e-3, "path {}", c.path_reflectance);
        assert!(c.transmittance_down > 0.99 && c.transmittance_up > 0.99);
        assert!(
            c.spherical_albedo < 5e-3,
            "spherical {}",
            c.spherical_albedo
        );
        assert!((c.surface_from_toa(0.3) - 0.3).abs() < 5e-3);
    }

    #[test]
    fn the_view_run_is_decorrelated_from_the_sun_run_by_a_xor() {
        // At `seed = 0x9e37_79b9` the XOR sends the view run to stream 0, while both `&` and `|`
        // would leave it on the seed itself — so a literal-0 anchor is what tells the three apart.
        // Same μ for the two runs, so the stream is the only thing that can differ.
        let atm = Atmosphere {
            optical_depth: 0.7,
            single_scattering_albedo: 0.85,
            phase: PhaseFunction::HenyeyGreenstein { asymmetry: 0.1 },
        };
        let (mu, photons, seed) = (0.75, 4000, 0x9e37_79b9_u64);
        let c = coupling_from_slab(&atm, mu, mu, 1.5, photons, seed);
        let on_zero = monte_carlo_slab(&atm, mu, photons, 0).transmittance;
        let on_seed = monte_carlo_slab(&atm, mu, photons, seed).transmittance;
        assert_ne!(
            on_zero, on_seed,
            "the two streams must differ for the anchor to discriminate"
        );
        assert_eq!(c.transmittance_down, on_seed);
        assert_eq!(c.transmittance_up, on_zero);
    }

    #[test]
    fn the_spherical_albedo_is_the_flux_weighted_average_of_the_nodes_own_streams() {
        // ∫₀¹ 2μ dμ = 1: the rule's weights sum to one against the 2μ factor, which is what makes
        // the result a flux average rather than a plain mean of the four node reflectances.
        let norm: f64 = GAUSS_01.iter().map(|(mu, w)| w * 2.0 * mu).sum();
        assert!((norm - 1.0).abs() < 1e-15, "Σ wᵢ·2μᵢ = {norm}");

        let atm = Atmosphere {
            optical_depth: 1.2,
            single_scattering_albedo: 0.9,
            phase: PhaseFunction::HenyeyGreenstein { asymmetry: 0.0 },
        };
        let (photons, seed) = (4000u32, 7u64);
        let c = coupling_from_slab(&atm, 0.8, 0.6, 1.5, photons, seed);
        // The i-th node runs on `seed + 101 + i` — 108, 109, 110, 111 here, written out. A fixed
        // seed makes the Monte-Carlo deterministic, so the sum reproduces to the bit.
        let mut expect = 0.0;
        for (k, node_seed) in [108u64, 109, 110, 111].into_iter().enumerate() {
            let (mu, w) = GAUSS_01[k];
            expect += w * monte_carlo_slab(&atm, mu, photons, node_seed).reflectance * 2.0 * mu;
        }
        assert_eq!(c.spherical_albedo, expect);
    }

    #[test]
    fn brdf_kernels_match_the_reference_formulas() {
        // (θs, θv, φ) in degrees and (K_vol, K_geo) from an independent numpy implementation of the
        // RossThick and LiSparse-Reciprocal (h/b=2, b/r=1) kernels.
        let cases: [(f64, f64, f64, f64, f64); 5] = [
            (30.0, 0.0, 0.0, -0.031442896087683136, -0.6982224735605751),
            (45.0, 30.0, 60.0, 0.06123861247758755, -0.9552160455924881),
            (20.0, 40.0, 120.0, -0.08625877784687785, -1.224820988564084),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (60.0, 50.0, 180.0, 0.14135337292689554, -2.532088886237956),
        ];
        for (ts, tv, phi, kvol, kgeo) in cases {
            let (s, v, p) = (ts.to_radians(), tv.to_radians(), phi.to_radians());
            assert!(
                (ross_thick_kernel(s, v, p) - kvol).abs() < 1e-12,
                "Kvol {ts},{tv},{phi}"
            );
            assert!(
                (li_sparse_reciprocal_kernel(s, v, p) - kgeo).abs() < 1e-12,
                "Kgeo {ts},{tv},{phi}"
            );
            // Both kernels vanish at nadir-nadir, so the RTLSR sum has to be pinned off-nadir:
            // each kernel enters once, with its own weight and a plus sign.
            let rho = brdf_reflectance(0.15, 0.05, 0.02, s, v, p);
            let expect = 0.15 + 0.05 * kvol + 0.02 * kgeo;
            assert!(
                (rho - expect).abs() < 1e-12,
                "ρ {ts},{tv},{phi}: {rho} vs {expect}"
            );
        }
        // At nadir view and nadir sun both kernels vanish, so the BRDF is the isotropic term.
        assert!((brdf_reflectance(0.15, 0.05, 0.02, 0.0, 0.0, 0.0) - 0.15).abs() < 1e-12);
    }

    /// A uniform k-distribution gets exactly the equal split it got before, term for term.
    ///
    /// This is the compatibility statement the redistribution rests on: where the weights carry no
    /// information there is nothing to redistribute, and `m Σ wⱼ² = 1` means the proportional rule
    /// and the equal one are the same rule. If this drifts, every channel whose weights are near
    /// uniform has silently changed its sample size.
    #[test]
    fn uniform_weights_get_the_equal_split_they_always_got() {
        for terms in [1_usize, 2, 4, 256] {
            #[expect(clippy::cast_precision_loss, reason = "term counts are far below 2^53")]
            let weight = 1.0 / terms as f64;
            assert_eq!(term_photons(40_000, terms, weight), 40_000);
        }
    }

    /// The budget follows the weight, and the total it is drawn from does not move.
    ///
    /// Two claims in one: the split is proportional (a term holding half the weight of a
    /// `terms`-term distribution draws `terms/2` times the per-term average), and the sum over a
    /// distribution stays at `photons × terms`. The second is what makes "per term on average" an
    /// honest description of the parameter rather than a rename.
    #[test]
    fn the_split_is_proportional_and_conserves_the_total() {
        let weights = [0.5, 0.25, 0.125, 0.125];
        assert_eq!(term_photons(1_000, 4, 0.5), 2_000);
        assert_eq!(term_photons(1_000, 4, 0.25), 1_000);
        let total: u64 = weights
            .iter()
            .map(|&w| u64::from(term_photons(1_000, 4, w)))
            .sum();
        assert_eq!(total, 4_000);
    }

    /// A term too light to earn a photon gets one anyway, and a weight nobody can describe does too.
    ///
    /// Returning zero would drop the term's draw while leaving its weight in the sum, which is a
    /// silent bias rather than a loud failure — and a non-finite or negative weight reaching here
    /// means the caller skipped [`crate::gas::KDistribution`]'s validation, so the safe answer is the
    /// cheapest draw rather than a panic. The saturating end matters for the opposite reason: a
    /// weight far above 1 must clamp rather than wrap to a small count.
    #[test]
    fn a_negligible_or_undescribable_weight_still_draws_one_photon() {
        for weight in [0.0, 1.0e-9, -1.0, f64::NAN, f64::NEG_INFINITY] {
            assert_eq!(term_photons(40_000, 256, weight), 1, "weight {weight}");
        }
        assert_eq!(term_photons(u32::MAX, 256, 1.0), u32::MAX);
        assert_eq!(term_photons(40_000, 256, f64::INFINITY), u32::MAX);
    }

    /// A window channel comes out of the k-distribution path bit-identical to one that never entered it.
    ///
    /// 197 of RadCalNet's 211 wavelengths have no absorption in at least one gas, and the machinery
    /// that exists for the other 14 must not perturb them — not by a Monte-Carlo seed, and not by the
    /// last bit of a single-scattering albedo that was multiplied and divided by the same optical
    /// depth. Both spellings of "no gas" are checked: an empty term list, and the explicit
    /// unit-weight-at-zero-depth term the fixture actually writes.
    #[test]
    fn a_channel_with_no_absorber_is_the_plain_coupling_to_the_bit() {
        let atm = Atmosphere {
            optical_depth: 0.13,
            single_scattering_albedo: 0.88,
            phase: PhaseFunction::HenyeyGreenstein { asymmetry: 0.6 },
        };
        let (mu_sun, mu_view, azimuth, photons, seed) = (0.85, 0.95, 1.2, 20_000u32, 7u64);
        let plain = coupling_from_slab(&atm, mu_sun, mu_view, azimuth, photons, seed);
        for terms in [&[][..], &[(1.0, 0.0)][..]] {
            let correlated =
                correlated_k_coupling(&atm, terms, mu_sun, mu_view, azimuth, photons, seed);
            assert_eq!(correlated.terms().len(), 1);
            assert_eq!(correlated.terms()[0].0, 1.0);
            assert_eq!(correlated.terms()[0].1, plain);
        }
    }

    /// The inverse is the inverse, over the whole range a surface can occupy.
    ///
    /// There is no closed form once the terms differ — a sum of Möbius functions with different poles
    /// is a rational function of the term count — so the bisection is the definition and this is what
    /// says it converged. Run over negative reflectances too, because the retrieval reaches them
    /// whenever a scene is darker than the path reflectance and clamping there would hide the input
    /// error that put it there.
    #[test]
    fn the_correlated_k_inverse_undoes_the_forward_over_the_whole_bracket() {
        let atm = Atmosphere {
            optical_depth: 0.2,
            single_scattering_albedo: 0.9,
            phase: PhaseFunction::Rayleigh,
        };
        let terms = [(0.6, 0.0), (0.25, 0.35), (0.15, 4.0)];
        let coupling = correlated_k_coupling(&atm, &terms, 0.8, 0.9, 0.4, 20_000, 11);
        for step in -4..=9 {
            let surface = f64::from(step) * 0.1;
            let toa = coupling.toa_from_surface(surface);
            let back = coupling
                .surface_from_toa(toa)
                .expect("reachable by construction");
            assert!(
                (back - surface).abs() < 1e-12,
                "rho_s {surface} -> rho_TOA {toa} -> {back}"
            );
        }
        assert!(coupling.surface_from_toa(-1.0).is_none());
        assert!(coupling.surface_from_toa(2.0).is_none());
    }

    /// Averaging the couplings is not the same as weight-summing what they predict, and the gap is the
    /// whole reason the terms are carried.
    ///
    /// A band core and a band wing have wildly different `T↓T↑`, and `ρ_TOA` is not linear in the
    /// coupling: collapsing the k-distribution to its mean coupling and inverting once retrieves a
    /// different surface. Measured here rather than argued, on a two-term channel whose terms are
    /// transparent and nearly opaque — which is what a real band core and its wing are.
    #[test]
    fn one_averaged_coupling_is_not_the_k_distribution_and_the_gap_is_measured() {
        let atm = Atmosphere {
            optical_depth: 0.15,
            single_scattering_albedo: 0.92,
            phase: PhaseFunction::Rayleigh,
        };
        let terms = [(0.7, 0.0), (0.3, 3.0)];
        let coupling = correlated_k_coupling(&atm, &terms, 0.8, 0.9, 0.4, 40_000, 23);
        let mut mean = AtmosphericCoupling {
            path_reflectance: 0.0,
            transmittance_down: 0.0,
            transmittance_up: 0.0,
            spherical_albedo: 0.0,
        };
        for &(weight, term) in coupling.terms() {
            mean.path_reflectance += weight * term.path_reflectance;
            mean.transmittance_down += weight * term.transmittance_down;
            mean.transmittance_up += weight * term.transmittance_up;
            mean.spherical_albedo += weight * term.spherical_albedo;
        }
        let truth = 0.30;
        let toa = coupling.toa_from_surface(truth);
        let collapsed = mean.surface_from_toa(toa);
        let exact = coupling
            .surface_from_toa(toa)
            .expect("reachable by construction");
        println!(
            "MEASURED k-distribution collapse: rho_s {truth:.3} retrieved {exact:.4} exactly, \
             {collapsed:.4} from the mean coupling ({:.1} % away)",
            100.0 * (collapsed - truth) / truth
        );
        assert!((exact - truth).abs() < 1e-9);
        assert!(
            (collapsed - truth).abs() > 0.05 * truth,
            "the mean coupling retrieved {collapsed:.4} against {truth:.3}, which would make the \
             per-term couplings pointless"
        );
    }

    /// The absorber dilutes the scattering without changing it.
    ///
    /// `ω τ` is the scattering optical depth, and a gas that absorbs cannot create or destroy a
    /// scatterer. That is the invariant `with_absorber` exists to hold, and writing the two fields by
    /// hand is where it gets dropped.
    #[test]
    fn adding_an_absorber_holds_the_scattering_optical_depth_fixed() {
        let atm = Atmosphere {
            optical_depth: 0.37,
            single_scattering_albedo: 0.81,
            phase: PhaseFunction::Rayleigh,
        };
        let scattering = atm.optical_depth * atm.single_scattering_albedo;
        for absorber in [0.0, 1e-6, 0.4, 12.0] {
            let mixed = atm.with_absorber(absorber);
            assert!(
                (mixed.optical_depth * mixed.single_scattering_albedo - scattering).abs() < 1e-15,
                "absorber {absorber} moved the scattering depth"
            );
            assert!((mixed.optical_depth - atm.optical_depth - absorber).abs() < 1e-15);
        }
        assert_eq!(
            atm.with_absorber(0.0).single_scattering_albedo,
            atm.single_scattering_albedo
        );
    }
}
