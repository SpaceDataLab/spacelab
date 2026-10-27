//! Monte Carlo radiative transfer through a plane-parallel atmosphere (Marchuk's method).
//!
//! A collimated solar beam enters the top of a homogeneous scattering-and-absorbing layer. Each
//! photon is traced by sampling exponential free paths in optical depth, absorbing with probability
//! `1 − ω` at each interaction and otherwise scattering into a new direction drawn from the layer's
//! [`PhaseFunction`] — analytic, or tabulated from Mie theory or an ice-crystal ray trace, which is
//! what lets a cirrus layer scatter like cirrus. Tallying where photons leave gives the **reflectance**
//! (back to space), **transmittance** (down to the surface) and **absorptance** — the path-radiance
//! core of atmospheric correction. Deterministic: a seeded `xoshiro256**` stream.
//!
//! Gate: `derivations/slab_single_scattering.mac` derives the next-event estimator's
//! first-collision expectation — the single-scattering reflectance, with the module's `2` against
//! the literature's `4` — its hemispheric integral as the first-order plane albedo, the direct
//! beam, the two scattering rotations as the spherical law of cosines and a unit vector at `cos θ`,
//! and the scattering depth `with_absorber` holds fixed; `derivations/slab_single_scattering.gp`
//! redoes the integrals by quadrature. The module test
//! `the_first_collision_estimate_has_the_single_scattering_expectation` (G2) holds the shipped
//! estimator's mean to that file's 40-digit value.

// Photon counters cast to f64 for the final fractions (values ≤ photon count, exact well past 2⁵³);
// the single-letter locals (μ, τ, g, s, …) are the notation of radiative transfer.
#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use crate::phase::PhaseFunction;

const TAU: f64 = core::f64::consts::TAU;
/// A high cap so a conservative-scattering photon can never loop forever; hit essentially never.
const MAX_SCATTERS: u32 = 100_000;

/// A homogeneous plane-parallel atmosphere layer.
///
/// The lifetime is the phase function's: a [`PhaseFunction::Tabulated`] borrows its grids from the
/// caller, so nothing here allocates and the layer stays `Copy`.
#[derive(Clone, Copy, Debug)]
pub struct Atmosphere<'a> {
    /// Total vertical optical depth `τ`.
    pub optical_depth: f64,
    /// Single-scattering albedo `ω` — the scatter-vs-absorb probability at an interaction, in `[0, 1]`.
    pub single_scattering_albedo: f64,
    /// What a scattering event does to the photon's direction.
    pub phase: PhaseFunction<'a>,
}

impl<'a> Atmosphere<'a> {
    /// The same layer with a purely absorbing component of vertical optical depth `absorber` mixed
    /// into it — one term of a gaseous k-distribution ([`crate::gas`]).
    ///
    /// The depth adds and the single-scattering albedo falls, because what the absorber cannot change
    /// is how much *scattering* there is: `ω τ`, the scattering optical depth, is held fixed and `ω`
    /// is whatever that makes it. Writing the two fields by hand is the same arithmetic and one of the
    /// two is easy to forget, which is a slab that scatters more the more it absorbs.
    ///
    /// Mixed in, not stacked on top. A gas in a separate layer above the aerosol would be a two-way
    /// `exp(−τ_g/μ)` factored out of the transport — honest for ozone at 25 km and wrong for water in
    /// the boundary layer, which is in among the particles it must not be factored out of.
    #[must_use]
    pub fn with_absorber(&self, absorber: f64) -> Atmosphere<'a> {
        if self.optical_depth <= 0.0 {
            // Nothing to dilute. A layer with no scattering scatters none of what it absorbs, and
            // with no absorption either it is not a layer at all and its albedo is unobservable.
            return Atmosphere {
                optical_depth: absorber,
                single_scattering_albedo: 0.0,
                phase: self.phase,
            };
        }
        Atmosphere {
            optical_depth: self.optical_depth + absorber,
            // Divided, not multiplied out. `ω/(1 + a/τ)` is *exactly* `ω` at `a = 0`, where the
            // algebraically identical `ω τ/(τ + a)` is a rounding away from it — and a window
            // channel has to come out of the k-distribution path bit-identical to one that never
            // entered it, or 197 of RadCalNet's 211 wavelengths get perturbed by the machinery
            // that exists for the other 14.
            single_scattering_albedo: self.single_scattering_albedo
                / (1.0 + absorber / self.optical_depth),
            phase: self.phase,
        }
    }

    /// This layer as the single slice of a stack, for the stratified transport.
    #[must_use]
    pub fn layer(&self) -> Layer<'a> {
        Layer {
            thickness: self.optical_depth,
            single_scattering_albedo: self.single_scattering_albedo,
            phase: self.phase,
        }
    }
}

/// The most slices a stratified stack may carry.
///
/// The walk needs the cumulative optical depth at every boundary, and this module allocates
/// nothing, so that running sum lives on the call stack and its length is fixed here. Sixty-four is
/// far above what the vertical structure of a plane-parallel column resolves — the arrangement
/// measurement this type exists for converged by twenty-four — and a longer stack is an assertion
/// rather than a silent truncation.
pub const MAX_LAYERS: usize = 64;

/// One slice of a vertically stratified atmosphere.
///
/// A stack is ordered **top down**: `layers[0]` is the slice sunlight enters first. Each slice
/// carries its own optical thickness, single-scattering albedo and phase function, which is what
/// lets Rayleigh and aerosol sit on their own vertical profiles instead of being averaged into one
/// effective medium — the approximation an [`Atmosphere`] makes.
///
/// A one-slice stack **is** an [`Atmosphere`]: [`monte_carlo_slab`] and [`path_reflectance`] are
/// [`monte_carlo_stack`] and [`stack_path_reflectance`] over `[atmosphere.layer()]`, through the
/// same kernel, bit for bit. Optical depth is additive along a path, so a boundary crossing costs no
/// extra random number and needs no special case; that is why the identity is exact rather than
/// statistical, and why a homogeneous medium cut into a dozen slices reproduces the undivided one.
#[derive(Clone, Copy, Debug)]
pub struct Layer<'a> {
    /// This slice's own vertical optical depth — a thickness, not a cumulative depth.
    pub thickness: f64,
    /// Single-scattering albedo `ω` inside this slice.
    pub single_scattering_albedo: f64,
    /// What a scattering event inside this slice does to the photon's direction.
    pub phase: PhaseFunction<'a>,
}

/// The energy budget of a slab for a collimated incident beam; the three fractions sum to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlabResult {
    /// Fraction of incident flux reflected back through the top.
    pub reflectance: f64,
    /// Fraction transmitted through the bottom (reaching the surface).
    pub transmittance: f64,
    /// Fraction absorbed within the layer.
    pub absorptance: f64,
    /// Fraction of photons still inside the layer when the per-photon scatter budget ran out, and
    /// so of no determined fate. Zero for every atmosphere thin enough to be resolved; it grows
    /// with `τ²` at `ω = 1`, reaching 2.8e-3 at `τ = 400`, where the walk needs more than
    /// `MAX_SCATTERS` steps to reach either boundary. Counting these as absorbed — which taking
    /// `1 − R − T` does — reports absorption in an atmosphere that cannot absorb.
    pub unresolved: f64,
}

/// A seeded `xoshiro256**` generator (seeded through `splitmix64`) — high quality, tiny, no `std`.
///
/// Shared with [`crate::ice`], which needs the same stream discipline and has no reason to carry a
/// second copy of it.
pub(crate) struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut splitmix = || {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^ (x >> 31)
        };
        Self {
            s: [splitmix(), splitmix(), splitmix(), splitmix()],
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`.
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0) // 2⁵³
    }

    /// Uniform in `(0, 1]` — safe for `−ln(ξ)`.
    pub(crate) fn open_unit(&mut self) -> f64 {
        1.0 - self.unit()
    }

    /// Uniform azimuth in `[0, 2π)`.
    pub(crate) fn azimuth(&mut self) -> f64 {
        TAU * self.unit()
    }
}

/// The vertical direction cosine after a photon travelling with cosine `mu` scatters through polar
/// angle `θ` (`cos_t = cos θ`, measured from its own direction) at azimuth `phi` about it.
fn scatter_mu(mu: f64, cos_t: f64, phi: f64) -> f64 {
    let sin_t = libm::sqrt((1.0 - cos_t * cos_t).max(0.0));
    (mu * cos_t + libm::sqrt((1.0 - mu * mu).max(0.0)) * sin_t * libm::cos(phi)).clamp(-1.0, 1.0)
}

/// Trace `photons` through the `atmosphere` for a collimated beam at `mu0 = cos(solar zenith)` and
/// return the reflectance / transmittance / absorptance. The surface is black (a transmitted photon
/// is one that reaches it). `seed` fixes the stream, so the result is reproducible.
///
/// # Panics
///
/// If `mu0` is not in `(0, 1]`.
#[must_use]
pub fn monte_carlo_slab(atmosphere: &Atmosphere, mu0: f64, photons: u32, seed: u64) -> SlabResult {
    assert!(
        mu0 > 0.0 && mu0 <= 1.0,
        "mu0 = cos(solar zenith) must be in (0, 1]"
    );
    let layers = [atmosphere.layer()];
    let (edge, used) = edges(&layers);
    walk(&layers, &edge[..used], mu0, photons, seed)
}

/// [`monte_carlo_slab`] over a vertically stratified stack, ordered top down.
///
/// The free path is sampled in *optical* depth, which is additive across slices, so crossing a
/// boundary costs no extra random number: only which slice's `ω` and phase function the next
/// collision reads changes. A homogeneous stack therefore reproduces [`monte_carlo_slab`] bit for
/// bit however finely it is cut, which is the check that says a layer boundary introduces nothing
/// of its own.
///
/// # Panics
///
/// If `mu0` is not in `(0, 1]`, or the stack is empty or longer than [`MAX_LAYERS`].
#[must_use]
pub fn monte_carlo_stack(layers: &[Layer], mu0: f64, photons: u32, seed: u64) -> SlabResult {
    assert!(
        mu0 > 0.0 && mu0 <= 1.0,
        "mu0 = cos(solar zenith) must be in (0, 1]"
    );
    let (edge, used) = edges(layers);
    walk(layers, &edge[..used], mu0, photons, seed)
}

/// Cumulative optical depth at every slice boundary: `edges[0]` is 0 and `edges[layers.len()]` is
/// the stack's total depth. The second return is how much of the fixed buffer is in use.
fn edges(layers: &[Layer]) -> ([f64; MAX_LAYERS + 1], usize) {
    assert!(
        !layers.is_empty() && layers.len() <= MAX_LAYERS,
        "a stack carries 1 to {MAX_LAYERS} slices"
    );
    let mut out = [0.0; MAX_LAYERS + 1];
    // Summed from the first thickness rather than from a leading zero, so a one-slice stack's total
    // is that slice's depth *exactly* and not `0.0 + depth`. The two agree for every depth a caller
    // can mean; starting here leaves nothing to agree about, and is what lets `monte_carlo_slab`
    // share this kernel with no bit of its shipped answer riding on the arithmetic.
    out[1] = layers[0].thickness;
    for (index, layer) in layers.iter().enumerate().skip(1) {
        // Not an `assert!`: `monte_carlo_slab` reaches the same kernel and has always accepted
        // whatever depth it was handed, so promoting this to a release check would narrow a shipped
        // function's domain rather than a new one's. Ascending edges are what `slice_at`'s bisection
        // needs, and a stack that violates them is a caller error every test build catches.
        debug_assert!(
            layer.thickness >= 0.0,
            "slice {index} has a negative optical thickness"
        );
        out[index + 1] = out[index] + layer.thickness;
    }
    (out, layers.len() + 1)
}

/// Which slice the vertical optical depth `tz` sits in — `edges[i] <= tz < edges[i + 1]`.
///
/// Bisected rather than scanned: this runs once per collision, and a linear scan would make a
/// stratified atmosphere's cost a function of how finely it is sliced, which is the variable a
/// stack exists to vary.
fn slice_at(edges: &[f64], tz: f64) -> usize {
    edges.partition_point(|&edge| edge <= tz).max(1) - 1
}

/// The analog walk, shared by [`monte_carlo_slab`] and [`monte_carlo_stack`]. `edges` is `layers`'
/// cumulative depth, so `edges[layers.len()]` is the surface.
fn walk(layers: &[Layer], edges: &[f64], mu0: f64, photons: u32, seed: u64) -> SlabResult {
    let tau = edges[edges.len() - 1];
    let mut rng = Rng::new(seed);
    let (mut reflected, mut transmitted, mut absorbed) = (0u64, 0u64, 0u64);
    for _ in 0..photons {
        let mut tz = 0.0_f64; // vertical optical depth: 0 at the top, τ at the bottom
        let mut mu = mu0; // vertical direction cosine, > 0 downward
        for _ in 0..MAX_SCATTERS {
            tz += -libm::log(rng.open_unit()) * mu; // free path, projected onto the vertical
            if tz <= 0.0 {
                reflected += 1;
                break;
            }
            if tz >= tau {
                transmitted += 1;
                break;
            }
            let here = &layers[slice_at(edges, tz)];
            if rng.unit() >= here.single_scattering_albedo {
                absorbed += 1;
                break;
            }
            let cos_t = here.phase.sample(rng.unit());
            mu = scatter_mu(mu, cos_t, rng.azimuth());
        }
    }
    let n = f64::from(photons);
    SlabResult {
        reflectance: reflected as f64 / n,
        transmittance: transmitted as f64 / n,
        absorptance: absorbed as f64 / n,
        unresolved: (u64::from(photons) - reflected - transmitted - absorbed) as f64 / n,
    }
}

/// The direction a photon travels after scattering through `cos_t` at azimuth `phi` about `d`.
///
/// The three-dimensional form of [`scatter_mu`], which it agrees with exactly on the vertical
/// component: `−z` here is `scatter_mu`'s `μ`, term for term. A local estimator needs the other two
/// components as well, because the angle to the *sensor* is not a function of the vertical alone.
fn scatter_direction(d: [f64; 3], cos_t: f64, phi: f64) -> [f64; 3] {
    let sin_t = libm::sqrt((1.0 - cos_t * cos_t).max(0.0));
    let (cos_p, sin_p) = (libm::cos(phi), libm::sin(phi));
    let horizontal = libm::sqrt((1.0 - d[2] * d[2]).max(0.0));
    // At the poles the azimuthal frame about `d` degenerates: `φ` is measured from a reference
    // direction that does not exist there, and the general expression divides by a horizontal
    // component that is exactly zero. This is not a limit of the general case — approaching `−ẑ`
    // along `+x` and along `+y` give different answers, which is what "degenerate" means — but a
    // choice of reference, and the global frame is the natural one. What the caller needs is what
    // survives the choice: a unit direction at polar angle `θ` from `d`, with `φ` sweeping a full
    // circle about it, and that holds here as everywhere else.
    if horizontal < 1e-9 {
        let sign = if d[2] >= 0.0 { 1.0 } else { -1.0 };
        return [sin_t * cos_p, sin_t * sin_p, sign * cos_t];
    }
    let k = sin_t / horizontal;
    [
        k * (d[0] * d[2] * cos_p - d[1] * sin_p) + d[0] * cos_t,
        k * (d[1] * d[2] * cos_p + d[0] * sin_p) + d[1] * cos_t,
        -horizontal * sin_t * cos_p + d[2] * cos_t,
    ]
}

/// The **bidirectional** path reflectance of the slab over a black surface — 6S's `ρ_atm`.
///
/// [`monte_carlo_slab`]'s `reflectance` is the plane albedo: the fraction of incident flux that
/// leaves through the top, in *any* direction. That is not the quantity the atmospheric-correction
/// coupling `ρ_TOA = ρ_atm + T↓T↑ρ_s/(1 − sρ_s)` calls `ρ_atm`, which is the reflectance a sensor
/// looking along one direction measures — a function of the relative azimuth as well as the two
/// zenith angles, and for a Rayleigh atmosphere tens of per cent away from its own hemispheric mean.
///
/// Estimated by **next-event estimation** rather than by binning exit directions: at every collision
/// the photon contributes the probability of scattering straight into the view direction, times the
/// transmission from there to the top,
///
/// ```text
///     Δρ = ω · p(cos Θ) · exp(−t/μ_v) / (2 μ_v),
/// ```
///
/// with `Θ` the angle between the photon's own direction and the view direction. Every collision
/// contributes to every direction, so the variance is a small fraction of what a directional bin —
/// which throws away all but the photons that happen to leave inside it — would give for the same
/// photon count. The `2` is this module's `∫₋₁¹ p dμ = 1` normalisation rather than a fudge: with the
/// `(1/4π)∫P dΩ = 1` convention of the transfer literature, `P = 2p` and the factor is the
/// textbook `4`.
///
/// # Geometry
///
/// `mu_sun` and `mu_view` are the cosines of the solar and view zenith angles, both positive.
/// `relative_azimuth` is measured from the *forward-scattering* half-plane: at `0` the sensor looks
/// along the beam's horizontal heading, and at `π` it looks back towards the sun, which is where
/// `mu_sun == mu_view` gives exact backscatter.
///
/// # Panics
///
/// If either cosine is outside `(0, 1]`.
#[must_use]
pub fn path_reflectance(
    atmosphere: &Atmosphere,
    mu_sun: f64,
    mu_view: f64,
    relative_azimuth: f64,
    photons: u32,
    seed: u64,
) -> f64 {
    assert!(
        mu_sun > 0.0 && mu_sun <= 1.0 && mu_view > 0.0 && mu_view <= 1.0,
        "both direction cosines must be in (0, 1]"
    );
    let layers = [atmosphere.layer()];
    let (edge, used) = edges(&layers);
    estimate(
        &layers,
        &edge[..used],
        mu_sun,
        mu_view,
        relative_azimuth,
        photons,
        seed,
    )
}

/// [`path_reflectance`] over a vertically stratified stack, ordered top down — same next-event
/// estimator, same draw order, so a homogeneous stack reproduces it bit for bit.
///
/// # Panics
///
/// If either cosine is outside `(0, 1]`, or the stack is empty or longer than [`MAX_LAYERS`].
#[must_use]
pub fn stack_path_reflectance(
    layers: &[Layer],
    mu_sun: f64,
    mu_view: f64,
    relative_azimuth: f64,
    photons: u32,
    seed: u64,
) -> f64 {
    assert!(
        mu_sun > 0.0 && mu_sun <= 1.0 && mu_view > 0.0 && mu_view <= 1.0,
        "both direction cosines must be in (0, 1]"
    );
    let (edge, used) = edges(layers);
    estimate(
        layers,
        &edge[..used],
        mu_sun,
        mu_view,
        relative_azimuth,
        photons,
        seed,
    )
}

/// The next-event estimator, shared by [`path_reflectance`] and [`stack_path_reflectance`].
fn estimate(
    layers: &[Layer],
    edges: &[f64],
    mu_sun: f64,
    mu_view: f64,
    relative_azimuth: f64,
    photons: u32,
    seed: u64,
) -> f64 {
    let tau = edges[edges.len() - 1];

    let sun = [libm::sqrt((1.0 - mu_sun * mu_sun).max(0.0)), 0.0, -mu_sun];
    let sin_view = libm::sqrt((1.0 - mu_view * mu_view).max(0.0));
    let view = [
        sin_view * libm::cos(relative_azimuth),
        sin_view * libm::sin(relative_azimuth),
        mu_view,
    ];

    let mut rng = Rng::new(seed);
    let mut total = 0.0;
    for _ in 0..photons {
        let mut d = sun;
        let mut tz = 0.0_f64;
        for _ in 0..MAX_SCATTERS {
            tz += -libm::log(rng.open_unit()) * -d[2];
            if tz <= 0.0 || tz >= tau {
                break; // out of the top, or through to the surface
            }
            let here = &layers[slice_at(edges, tz)];
            let omega = here.single_scattering_albedo;
            let cos_scatter = d[0] * view[0] + d[1] * view[1] + d[2] * view[2];
            total +=
                omega * here.phase.value(cos_scatter.clamp(-1.0, 1.0)) * libm::exp(-tz / mu_view)
                    / (2.0 * mu_view);
            if rng.unit() >= omega {
                break; // absorbed
            }
            d = scatter_direction(d, here.phase.sample(rng.unit()), rng.azimuth());
        }
    }
    total / f64::from(photons)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::excessive_precision,
        clippy::unreadable_literal
    )]
    use super::*;

    const PI: f64 = core::f64::consts::PI;

    fn slab(tau: f64, omega: f64, g: f64) -> Atmosphere<'static> {
        Atmosphere {
            optical_depth: tau,
            single_scattering_albedo: omega,
            phase: PhaseFunction::HenyeyGreenstein { asymmetry: g },
        }
    }

    /// A uniform medium cut into slices must reproduce the undivided one **bit for bit**.
    ///
    /// Not a tolerance, and not a statistical agreement: optical depth is additive along a path, so
    /// slicing a uniform medium changes nothing the walk reads. The free path is drawn from the same
    /// number, in the same order; only *which* slice supplies `ω` and the phase function moves, and
    /// with no contrast across the boundary that lookup returns the same medium. That is what lets
    /// [`monte_carlo_slab`] route through the stack kernel instead of carrying a second copy of the
    /// walk — the shipped single-layer answer is a special case of the stack, not a parallel
    /// implementation that has to be kept in step by hand.
    ///
    /// The two cases are not equally strong, and they are not true for the same reason. One slice is
    /// **structural**: `edges` starts its prefix sum at the first thickness rather than at a leading
    /// zero, so the total is that slice's depth exactly and the kernel cannot tell the two apart.
    /// Twelve is **measured**: `0.24 / 12.0` summed twelve times lands one ulp below `0.24`, so the
    /// stack's `tau` is not the layer's, and the runs agree anyway because no free path drawn here
    /// ever falls in that one-ulp window. A different depth or slice count could in principle land
    /// in it; this asserts that these do not, which is the honest form of the claim and a more
    /// informative one than picking a depth that divides exactly.
    #[test]
    fn a_uniform_stack_reproduces_the_undivided_layer() {
        let (tau, omega, g) = (0.37, 0.93, 0.42);
        let one = slab(tau, omega, g);
        let (mu_sun, mu_view, photons, seed) = (0.71, 0.86, 20_000u32, 20_260_821u64);

        for slices in [1_usize, 12] {
            let cut: [Layer; 12] = core::array::from_fn(|_| Layer {
                thickness: tau / slices as f64,
                single_scattering_albedo: omega,
                phase: one.phase,
            });
            let stack = &cut[..slices];

            assert_eq!(
                monte_carlo_slab(&one, mu_sun, photons, seed),
                monte_carlo_stack(stack, mu_sun, photons, seed),
                "{slices} uniform slices must reproduce monte_carlo_slab"
            );
            let undivided = path_reflectance(&one, mu_sun, mu_view, 0.0, photons, seed);
            let sliced = stack_path_reflectance(stack, mu_sun, mu_view, 0.0, photons, seed);
            assert!(
                undivided.to_bits() == sliced.to_bits(),
                "{slices} uniform slices must reproduce path_reflectance: \
                 {undivided:.17e} against {sliced:.17e}"
            );
        }
    }

    /// And the stack must *resolve* contrast it is given, or the slice lookup is decorative.
    ///
    /// The same two components in the same total depth, swapped top for bottom. A plane-parallel
    /// atmosphere is not symmetric under that swap — the upper layer is what a photon meets first
    /// on the way in and last on the way out — so the two orders have to disagree.
    ///
    /// **The swap alone does not pin the lookup, and the mutation block is what said so.** A walk
    /// that reads slice zero and never looks again still reports two different answers for the two
    /// orders, because slice zero is a different layer in each; `slice_at -> 0` survived this test
    /// in exactly that way. What such a walk cannot report is an absorber it never reads, so the
    /// second case buries `ω = 0` in the middle of an otherwise conservative stack. A correct walk
    /// absorbs there; a walk that always reads the top sees `ω = 1` everywhere, and
    /// `rng.unit() >= 1.0` never holds, so its absorptance is **exactly** zero. Measured: 0.527210
    /// against 0.000000. That is a kill and not a tolerance, which is the difference between
    /// asserting the lookup happens and asserting that two numbers differ.
    #[test]
    fn the_order_of_two_unlike_slices_changes_the_answer() {
        let thin = Layer {
            thickness: 0.15,
            single_scattering_albedo: 1.0,
            phase: PhaseFunction::Rayleigh,
        };
        let thick = Layer {
            thickness: 0.35,
            single_scattering_albedo: 0.85,
            phase: PhaseFunction::HenyeyGreenstein { asymmetry: 0.7 },
        };
        let (mu0, photons, seed) = (0.6, 200_000u32, 20_260_821u64);

        let over = monte_carlo_stack(&[thin, thick], mu0, photons, seed).reflectance;
        let under = monte_carlo_stack(&[thick, thin], mu0, photons, seed).reflectance;
        // 200 000 photons put the Monte-Carlo error on a reflectance of order 0.1 near 7e-4; the
        // separation asserted is thirty times that, so this is a fact about the ordering and not
        // about the seed.
        assert!(
            libm::fabs(over - under) > 0.02,
            "the two orders must disagree: {over:.6} against {under:.6}"
        );

        let clear = |thickness| Layer {
            thickness,
            single_scattering_albedo: 1.0,
            phase: PhaseFunction::Rayleigh,
        };
        let opaque = Layer {
            thickness: 0.5,
            single_scattering_albedo: 0.0,
            phase: PhaseFunction::Rayleigh,
        };
        let buried = monte_carlo_stack(&[clear(0.2), opaque, clear(0.2)], mu0, photons, seed);
        assert!(
            buried.absorptance > 0.2,
            "an absorber buried mid-stack must absorb; reading slice zero forever reports \
             exactly none: {:.6}",
            buried.absorptance
        );
    }

    /// The single-scattering limit, where the path reflectance has a closed form — and the size of
    /// what is left over, which identifies itself.
    ///
    /// `ρ¹ = ω p(cos Θ) (1 − e^{−τ(1/μ_s + 1/μ_v)}) / (2(μ_s + μ_v))` is the collision-density
    /// integral done exactly, and the estimator must sit just above it: the excess is the scattering
    /// orders the closed form omits. A tolerance alone would not say *which* — so the test measures
    /// the excess at two single-scattering albedos. Second-order scattering carries `ω²` against the
    /// first order's `ω`, so a tenfold `ω` must multiply the excess by ten and leave the first-order
    /// term alone. Measured: 1.82 % excess at `ω = 0.1` and 18.14 % at `ω = 1`, a ratio of 9.97.
    ///
    /// A wrong constant in the estimator would be a fixed factor instead, identical at both.
    #[test]
    fn the_path_reflectance_matches_the_single_scattering_law_where_that_is_the_answer() {
        let (mu_sun, mu_view, azimuth, tau) = (0.8_f64, 0.6_f64, 2.0_f64, 0.05_f64);
        let sin_sun = (1.0 - mu_sun * mu_sun).sqrt();
        let sin_view = (1.0 - mu_view * mu_view).sqrt();
        let cos_scatter = sin_sun * sin_view * libm::cos(azimuth) - mu_sun * mu_view;

        let excess = |omega: f64| {
            let atmosphere = slab(tau, omega, 0.3);
            let got =
                path_reflectance(&atmosphere, mu_sun, mu_view, azimuth, 4_000_000, 20_260_815);
            let want = omega
                * atmosphere.phase.value(cos_scatter)
                * (1.0 - libm::exp(-tau * (1.0 / mu_sun + 1.0 / mu_view)))
                / (2.0 * (mu_sun + mu_view));
            got / want - 1.0
        };
        let (weak, strong) = (excess(0.1), excess(1.0));
        assert!(
            weak > 0.0 && weak < 0.03,
            "at omega = 0.1 the estimator is {weak} above the single-scattering law"
        );
        assert!(
            (strong / weak / 10.0 - 1.0).abs() < 0.15,
            "the excess went {weak} to {strong} for a tenfold omega; second order would be tenfold"
        );
    }

    /// (G2) The next-event estimator's first-collision term has the single-scattering expectation.
    ///
    /// `derivations/slab_single_scattering.mac` integrates the estimator's first-collision term
    /// `ω p(cos Θ) e^{−t/μᵥ}/(2μᵥ)` against the collision-depth density `e^{−t/μₛ}/μₛ` and gets the
    /// closed form `ρ¹ = ω p (1 − e^{−τ(1/μₛ + 1/μᵥ)})/(2(μₛ + μᵥ))`, which is the transfer
    /// literature's `π I/(μₛ F)` with `P = 2p`; `derivations/slab_single_scattering.gp` redoes the
    /// integral by quadrature. At `ω → 0` the later collisions carry `O(ω)` of the answer and drop
    /// out, so the Monte Carlo mean divided by `ω` must sit on that file's 40-digit `ρ¹/ω` to within
    /// its sampling error alone — at `μₛ = 4/5`, `μᵥ = 3/5`, relative azimuth `π`, where
    /// `cos Θ = −24/25` exactly. The same file puts the per-photon relative standard deviation at
    /// 0.8016, so two million photons carry 5.7e-4; the tolerance is five of those.
    ///
    /// Unlike the test above this does not difference two runs: a wrong constant in the estimator,
    /// a wrong collision density or a wrong attenuation to the sensor all move the mean itself.
    #[test]
    fn the_first_collision_estimate_has_the_single_scattering_expectation() {
        const RHO1_OVER_OMEGA: f64 = 7.14792150410902061369450414069634477139e-2;
        let omega = 1e-6;
        let atmosphere = slab(1.0, omega, 0.3);
        let got = path_reflectance(&atmosphere, 0.8, 0.6, PI, 2_000_000, 20_261_027) / omega;
        assert!(
            (got / RHO1_OVER_OMEGA - 1.0).abs() < 2.8e-3,
            "first-collision estimate {got:e} against ρ¹/ω = {RHO1_OVER_OMEGA:e}"
        );
    }

    /// The two estimators are the same physics: the flux-weighted hemispheric integral of the
    /// directional reflectance is the plane albedo the photon count gives.
    ///
    /// `r(μ_s) = (1/π) ∫₀^{2π} ∫₀^1 ρ(μ, φ) μ dμ dφ`, and this holds at every scattering order, so
    /// unlike the closed form above it is not a thin-atmosphere check. It is also the only test that
    /// can catch the two disagreeing: `monte_carlo_slab` counts photons through the top and
    /// `path_reflectance` never lets one out, and the arithmetic that connects them is where a
    /// convention gets dropped.
    #[test]
    fn the_directional_reflectance_integrates_to_the_plane_albedo() {
        const AZIMUTHS: usize = 12;
        let atmosphere = slab(0.5, 0.9, 0.4);
        let mu_sun = 0.7;
        // Gauss–Legendre on (0, 1] for μ, midpoint on the azimuth — periodic, so it converges fast.
        let nodes = [
            (0.069_431_844_202_973_67, 0.173_927_422_568_726_9),
            (0.330_009_478_207_571_85, 0.326_072_577_431_273_05),
            (0.669_990_521_792_428_1, 0.326_072_577_431_273_05),
            (0.930_568_155_797_026_3, 0.173_927_422_568_726_9),
        ];
        let mut integral = 0.0;
        for (i, (mu, w)) in nodes.iter().enumerate() {
            for k in 0..AZIMUTHS {
                let phi = TAU * (k as f64 + 0.5) / AZIMUTHS as f64;
                let seed = 918_273 + (i * AZIMUTHS + k) as u64;
                integral += w
                    * (TAU / AZIMUTHS as f64)
                    * mu
                    * path_reflectance(&atmosphere, mu_sun, *mu, phi, 60_000, seed);
            }
        }
        let albedo = integral / PI;
        let flux = monte_carlo_slab(&atmosphere, mu_sun, 2_000_000, 4_242).reflectance;
        assert!(
            (albedo / flux - 1.0).abs() < 0.02,
            "hemispheric integral {albedo} against the plane albedo {flux}"
        );
    }

    /// The relative azimuth is read, and which way it leans is the phase function's to decide.
    ///
    /// At `μ_sun == μ_view` the two extremes of azimuth are the two extremes of scattering angle:
    /// `Δφ = π` is exactly backwards, `Δφ = 0` is a 74° deviation for this geometry. A forward-peaked
    /// Henyey–Greenstein is therefore brightest at `0`, and Rayleigh — whose `⅜(1 + μ²)` is *twice*
    /// as bright at 180° as at 90° — is brightest at `π`. The ordering reverses between the two, so
    /// no constant, and no azimuth-blind quantity, can produce both.
    #[test]
    fn the_azimuth_leans_the_way_the_phase_function_does() {
        let mu = 0.6;
        let forward = slab(0.3, 1.0, 0.75);
        let (f_ahead, f_back) = (
            path_reflectance(&forward, mu, mu, 0.0, 400_000, 11),
            path_reflectance(&forward, mu, mu, PI, 400_000, 11),
        );
        assert!(
            f_ahead > 1.5 * f_back,
            "a forward-peaked layer is brightest across the beam: {f_ahead} against {f_back}"
        );

        let rayleigh = Atmosphere {
            optical_depth: 0.3,
            single_scattering_albedo: 1.0,
            phase: PhaseFunction::Rayleigh,
        };
        let (r_ahead, r_back) = (
            path_reflectance(&rayleigh, mu, mu, 0.0, 400_000, 11),
            path_reflectance(&rayleigh, mu, mu, PI, 400_000, 11),
        );
        assert!(
            r_back > r_ahead,
            "Rayleigh is brightest straight back: {r_back} against {r_ahead}"
        );
    }

    #[test]
    fn every_photon_leaves_through_exactly_one_of_the_three_doors() {
        // ω = 0 fires the absorption test at the first interaction, so a photon either misses the
        // slab outright or is absorbed: A = 1 − exp(−τ/μ0), an external number rather than a
        // restatement of the census.
        for (tau, mu0) in [(1.0, 1.0), (1.0, 0.5), (2.0, 0.8)] {
            let r = monte_carlo_slab(&slab(tau, 0.0, 0.0), mu0, 300_000, 42);
            let expect = 1.0 - libm::exp(-tau / mu0);
            assert!(
                (r.absorptance - expect).abs() < 5e-3,
                "τ={tau} μ0={mu0}: A={} vs 1−exp(−τ/μ0)={expect}",
                r.absorptance
            );
        }
        // A scattering slab fills all three channels, so nothing in the census can hide behind a
        // count that happens to be zero.
        let r = monte_carlo_slab(&slab(1.5, 0.9, 0.2), 0.7, 200_000, 9);
        assert!(
            r.reflectance > 0.01 && r.transmittance > 0.01 && r.absorptance > 0.01,
            "R={} T={} A={}",
            r.reflectance,
            r.transmittance,
            r.absorptance
        );
        assert_eq!(r.unresolved, 0.0, "unresolved {}", r.unresolved);
        let census = r.reflectance + r.transmittance + r.absorptance;
        assert!((census - 1.0).abs() < 1e-12, "census {census}");
    }

    /// (G3) With `ω = 0` the slab is a Beer–Lambert absorber: transmittance `e^{−τ/μ₀}`, the
    /// direct-beam probability `derivations/slab_single_scattering.mac` integrates.
    #[test]
    fn pure_absorption_gives_beer_lambert_transmission() {
        // ω = 0: the only way through is to miss every interaction — direct transmission
        // exp(−τ/μ0), no reflection.
        for (tau, mu0) in [(1.0, 1.0), (1.0, 0.5), (2.0, 0.8)] {
            let r = monte_carlo_slab(&slab(tau, 0.0, 0.0), mu0, 300_000, 42);
            assert!(
                r.reflectance < 1e-9,
                "τ={tau} reflectance {}",
                r.reflectance
            );
            let expect = libm::exp(-tau / mu0);
            assert!(
                (r.transmittance - expect).abs() < 5e-3,
                "τ={tau} μ0={mu0}: T={} vs exp(−τ/μ0)={expect}",
                r.transmittance
            );
        }
    }

    /// (G4) A conservative slab sends every photon out through one of its two faces.
    #[test]
    fn conservative_scattering_conserves_energy() {
        // ω = 1: no absorption, so every photon leaves and R + T = 1 exactly. Six hundred photons,
        // not three hundred thousand: every assertion here is exact or has a margin of tens of
        // sigma, and the count is the only thing standing between a stalled walk and an assertion
        // failure — see `every_photon_leaves_a_conservative_slab_within_its_scatter_budget`.
        let r = monte_carlo_slab(&slab(2.0, 1.0, 0.5), 0.8, 600, 7);
        assert!(r.absorptance.abs() < 1e-6, "absorptance {}", r.absorptance);
        assert!((r.reflectance + r.transmittance - 1.0).abs() < 1e-6);
        assert!(r.reflectance > 0.0 && r.transmittance > 0.0);
    }

    /// Every photon leaves a conservative slab, and leaves it inside its scatter budget.
    ///
    /// `conservative_scattering_conserves_energy` asserts the same physics, but at 300 000 photons
    /// against a `MAX_SCATTERS` of 100 000. A mutation that stops the walk making vertical progress
    /// — a NaN direction cosine out of the phase function, a step of zero, a `mu` that decays
    /// geometrically — turns that into 3×10¹⁰ inner iterations, and the clock rather than the
    /// assertion is what notices: seven mutants of `open_unit`, `sample_hg_cos` and `scatter_mu`
    /// were recorded as timeouts for exactly that reason — an exhausted photon costs 7.4 ms in a
    /// debug build, so three hundred thousand of them is a day, not a test.
    ///
    /// A hundred photons per case cost at most 0.7 s and lose nothing, because with `omega = 1` the
    /// absorption branch is `xi >= 1.0`, which no sample in `[0, 1)` satisfies: absorptance is zero
    /// for *any* photon count, and a photon that exhausts its scatters — counted as neither
    /// reflected nor transmitted — is exactly what shows up in it. Both branches of the phase
    /// function are covered, since `g = 0` and `g != 0` take different ones.
    #[test]
    fn every_photon_leaves_a_conservative_slab_within_its_scatter_budget() {
        for (tau, mu0, g) in [(0.5, 1.0, 0.0), (2.0, 0.6, 0.7), (2.0, 0.6, -0.7)] {
            let r = monte_carlo_slab(&slab(tau, 1.0, g), mu0, 100, 12345);
            assert!(
                r.absorptance.abs() < 1e-12,
                "tau={tau} mu0={mu0} g={g}: {} of the beam never left",
                r.absorptance
            );
        }
    }

    #[test]
    fn thick_conservative_layer_is_highly_reflective() {
        // A thick conservative layer reflects most of the beam (semi-infinite ⇒ R → 1). Six hundred
        // photons put the true R ≈ 0.9 some twenty standard errors above the 0.6 cut, for the same
        // reason the test above runs at that count.
        let r = monte_carlo_slab(&slab(16.0, 1.0, 0.0), 1.0, 600, 99);
        assert!(r.reflectance > 0.6, "reflectance {}", r.reflectance);
        assert!((r.reflectance + r.transmittance - 1.0).abs() < 1e-6);
    }

    #[test]
    fn thin_layer_reflectance_scales_with_optical_depth() {
        // Single scattering dominates a thin layer, so reflectance is linear in τ.
        let r1 = monte_carlo_slab(&slab(0.05, 0.9, 0.0), 1.0, 600_000, 1).reflectance;
        let r2 = monte_carlo_slab(&slab(0.10, 0.9, 0.0), 1.0, 600_000, 1).reflectance;
        let ratio = r2 / r1;
        assert!((1.7..2.3).contains(&ratio), "R(0.1)/R(0.05) = {ratio}");
    }

    /// An atmosphere with `ω = 1` scatters every interaction and absorbs none, so its absorptance
    /// is exactly zero — not small, zero, since it is a count of a branch that cannot be taken.
    /// The census is what makes that expressible: taking absorptance as `1 − R − T` also collects
    /// every photon still walking when the scatter budget ran out, and a conservative slab thick
    /// enough to need more than `MAX_SCATTERS` steps then reports absorption it cannot have. At
    /// `τ = 400` that was 2.8e-3, forty times the 7e-5 sampling noise of these 20 000 photons.
    #[test]
    fn a_conservative_atmosphere_absorbs_nothing_however_thick_it_is() {
        for tau in [1.0_f64, 10.0, 100.0, 400.0] {
            let r = monte_carlo_slab(&slab(tau, 1.0, 0.0), 1.0, 20_000, 7);
            assert_eq!(r.absorptance, 0.0, "τ = {tau} absorbed {}", r.absorptance);
            // Every photon has exactly one fate, and `unresolved` is the one that is not physics.
            assert!(
                (r.reflectance + r.transmittance + r.absorptance + r.unresolved - 1.0).abs()
                    < 1e-12,
                "τ = {tau}: {r:?} does not account for every photon"
            );
        }
        // And it is reached: a thick conservative slab is what exhausts the budget.
        let thick = monte_carlo_slab(&slab(400.0, 1.0, 0.0), 1.0, 20_000, 7);
        assert!(
            thick.unresolved > 0.0,
            "τ = 400 resolved every photon, so the census has nothing to separate"
        );
    }

    #[test]
    fn same_seed_is_reproducible() {
        let a = monte_carlo_slab(&slab(1.0, 0.8, 0.3), 0.9, 50_000, 2024);
        let b = monte_carlo_slab(&slab(1.0, 0.8, 0.3), 0.9, 50_000, 2024);
        assert_eq!(a, b);
    }

    /// The generator against the published `xoshiro256**` stream.
    ///
    /// Every slab result is a Monte Carlo estimate with a statistical error of its own, so a
    /// degraded generator hides inside the tolerance: shifting the wrong way, or a state update
    /// update should XOR still yields random-looking numbers and moves the reflectance by less than
    /// the sampling noise. The stream is the one exactly specified thing about the generator, and
    /// the fixture comes from `tools/gen_xoshiro_oracle.py`, written from Blackman & Vigna's
    /// algorithm rather than from this code. Integer arithmetic throughout, so it is bit-exact.
    #[test]
    fn the_generator_reproduces_the_published_xoshiro256_stream() {
        const STREAM: [(u64, [u64; 4]); 3] = [
            (
                0,
                [
                    0x99ec_5f36_cb75_f2b4,
                    0xbf6e_1f78_4956_452a,
                    0x1a5f_849d_4933_e6e0,
                    0x6aa5_94f1_262d_2d2c,
                ],
            ),
            (
                1,
                [
                    0xb3f2_af6d_0fc7_10c5,
                    0x853b_5596_4736_4cea,
                    0x92f8_9756_082a_4514,
                    0x642e_1c7b_c266_a3a7,
                ],
            ),
            (
                42,
                [
                    0x1578_0b2e_0c2e_c716,
                    0x6104_d986_6d11_3a7e,
                    0xae17_5332_39e4_99a1,
                    0xecb8_ad47_03b3_60a1,
                ],
            ),
        ];
        for (seed, want) in STREAM {
            let mut rng = Rng::new(seed);
            for (i, w) in want.into_iter().enumerate() {
                assert_eq!(rng.next_u64(), w, "seed {seed}, draw {i}");
            }
        }
    }

    /// Every azimuth lies in one turn.
    ///
    /// `phi` only ever reaches the slab through `cos(phi)`, which is bounded and periodic, so a
    /// draw taken from the wrong interval still scatters plausibly and no reflectance can see it.
    /// The half-open range is the property: `TAU * unit()` is in `[0, 2π)`, and dividing, adding or
    /// subtracting instead all leave it.
    #[test]
    fn every_azimuth_is_one_turn() {
        let mut rng = Rng::new(2024);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for _ in 0..200_000 {
            let phi = rng.azimuth();
            assert!((0.0..TAU).contains(&phi), "azimuth {phi} outside [0, 2π)");
            lo = lo.min(phi);
            hi = hi.max(phi);
        }
        // ...and it reaches both ends, so the range assertion is not passing on a constant.
        assert!(lo < 1e-4 && hi > TAU - 1e-4, "azimuths span [{lo}, {hi}]");
    }

    /// Rodrigues' rotation of `v` about the unit axis `k` by `angle` — the reference the
    /// three-dimensional scattering is checked against, written from the formula rather than from
    /// the implementation, so that no term is shared between the two.
    fn rodrigues(v: [f64; 3], k: [f64; 3], angle: f64) -> [f64; 3] {
        let (c, s) = (libm::cos(angle), libm::sin(angle));
        let kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
        let cross = [
            k[1] * v[2] - k[2] * v[1],
            k[2] * v[0] - k[0] * v[2],
            k[0] * v[1] - k[1] * v[0],
        ];
        [0, 1, 2].map(|i| v[i] * c + cross[i] * s + k[i] * kv * (1.0 - c))
    }

    /// (G1) [`scatter_direction`] against a rotation built a different way.
    ///
    /// The slab never reads this function at all and the path estimator reads it only through a
    /// Monte Carlo average over a **uniform** azimuth, which is precisely the average that cannot
    /// see a transverse component: thirteen mutants of these five lines survived the seal, every one
    /// of them in a term the vertical cosine does not contain.
    ///
    /// The reference is the same rotation assembled from two independent pieces. Tilting `d` away
    /// from the zenith by `θ` means moving it along the unit vector in the `d`–`ẑ` plane
    /// perpendicular to `d`, which Gram–Schmidt gives as `u = (d(d·ẑ) − ẑ)/‖·‖`; turning that by the
    /// azimuth means rotating about `d` itself, which is Rodrigues. Neither step reuses a line of
    /// the closed form.
    #[test]
    fn a_scattering_event_is_the_rotation_it_is_written_as() {
        for d in [
            [0.6, 0.0, 0.8],
            [-0.36, 0.48, 0.8],
            [0.48, -0.36, -0.8],
            [0.8, 0.6, 0.0],
        ] {
            let u = {
                let raw = [d[0] * d[2], d[1] * d[2], d[2] * d[2] - 1.0];
                let n = libm::sqrt(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]);
                raw.map(|x| x / n)
            };
            for cos_t in [-0.9, -0.25, 0.0, 0.45, 0.99] {
                let sin_t = libm::sqrt(1.0 - cos_t * cos_t);
                let tilted = [0, 1, 2].map(|i| d[i] * cos_t + u[i] * sin_t);
                for k in 0..12 {
                    let phi = TAU * f64::from(k) / 12.0;
                    let got = scatter_direction(d, cos_t, phi);
                    let want = rodrigues(tilted, d, phi);
                    for i in 0..3 {
                        assert!(
                            (got[i] - want[i]).abs() < 1e-14,
                            "d={d:?} cosθ={cos_t} φ={phi}: {got:?} against {want:?}"
                        );
                    }
                }
            }
        }
    }

    /// At the poles the branch is a choice of reference, and what has to survive the choice.
    ///
    /// Straight up and straight down are the one place the general expression cannot be evaluated:
    /// `φ` is measured from a direction that does not exist there, and `horizontal` is exactly zero.
    /// It is tempting to justify the branch as the limit of the general case, and that is **false** —
    /// approaching `−ẑ` along `+x` sends the first component to `−sin θ cos φ` and approaching along
    /// `+y` sends it to `−sin θ sin φ`. A degenerate frame has no limit; it has conventions.
    ///
    /// So the test asserts what does not depend on the convention, which is also everything the
    /// caller uses: the result is a unit vector, it sits at polar angle `θ` from `d`, and `φ` turns
    /// its transverse part through a full circle at unit rate. A phase function integrates over `φ`,
    /// so a different reference is a relabelling of a variable that is about to be averaged away —
    /// but a wrong sign, a wrong product or a lost normalisation is not, and those are what this
    /// pins.
    ///
    /// This is the only test that reaches the branch at all: no photon in any slab run arrives
    /// exactly vertical, so the sign, both products and the guard itself were unexercised — ten of
    /// the seal's fourteen survivors are these five lines.
    #[test]
    fn at_the_poles_the_scattering_is_still_a_turn_through_the_angle_it_was_given() {
        for z in [1.0_f64, -1.0] {
            let d = [0.0, 0.0, z];
            for cos_t in [-0.8, 0.0, 0.35, 0.95] {
                let sin_t = libm::sqrt(1.0 - cos_t * cos_t);
                let reference = scatter_direction(d, cos_t, 0.0);
                for k in 0..8 {
                    let phi = TAU * f64::from(k) / 8.0;
                    let out = scatter_direction(d, cos_t, phi);
                    let norm = libm::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
                    let along = out[0] * d[0] + out[1] * d[1] + out[2] * d[2];
                    assert!(
                        (norm - 1.0).abs() < 1e-15,
                        "z={z} cosθ={cos_t} φ={phi}: |{out:?}|"
                    );
                    assert!(
                        (along - cos_t).abs() < 1e-15,
                        "z={z} cosθ={cos_t} φ={phi}: {out:?} lies at {along} from {d:?}"
                    );
                    // The transverse parts at φ and at 0 are two radii of the same circle, so their
                    // normalised inner product is cos φ — the statement that φ is an angle at all.
                    if sin_t > 0.0 {
                        let turned = (0..3)
                            .map(|i| (out[i] - cos_t * d[i]) * (reference[i] - cos_t * d[i]))
                            .sum::<f64>()
                            / (sin_t * sin_t);
                        assert!(
                            (turned - libm::cos(phi)).abs() < 1e-14,
                            "z={z} cosθ={cos_t} φ={phi}: turned by {turned}, wanted {}",
                            libm::cos(phi)
                        );
                    }
                }
            }
        }
    }

    /// The pole threshold sits in a gap of the values that can reach it.
    ///
    /// `horizontal` is `sqrt(1 - d_z^2)`, and near `|d_z| = 1` the spacing of `f64` is `2^-53`, so
    /// the smallest non-zero `1 - d_z^2` is `2^-52` and its square root is 1.49e-8 — fifteen times
    /// the 1e-9 threshold. Every direction is therefore either exactly on the pole or well clear of
    /// it, and nothing can land in between. That is why the comparison at the threshold is written
    /// with `<` rather than `<=` without the choice mattering, and it is the evidence behind the
    /// `.cargo/mutants.toml` entry for it: the two forms differ only on values that do not exist.
    ///
    /// The distinction is worth proving rather than assuming, because the branches do **not** agree
    /// where they meet — at `d_z < 0` the general expression tends to `-sin θ cos φ` in `x` where the
    /// pole branch returns `+sin θ cos φ`, which is the degeneracy the branch exists to name. A
    /// threshold that were reachable would be a real fork, not a rounding question.
    #[test]
    fn the_pole_threshold_is_unreachable_from_below() {
        let mut smallest = f64::INFINITY;
        for step in 1..64_u64 {
            for sign in [1.0_f64, -1.0] {
                let d_z = sign * f64::from_bits(1.0_f64.to_bits() - step);
                let horizontal = libm::sqrt((1.0 - d_z * d_z).max(0.0));
                if horizontal > 0.0 {
                    smallest = smallest.min(horizontal);
                }
            }
        }
        assert!(
            smallest > 1.0e-8,
            "a direction reaches horizontal = {smallest:.3e}, so the 1e-9 threshold is no longer \
             in a gap and which comparison stands at it becomes observable"
        );
    }

    /// Scattering commutes with the mirror through the plane the sun lies in.
    ///
    /// `path_reflectance` builds the beam as `[sin θ_s, 0, −μ_s]`, so the whole photon ensemble is
    /// symmetric about `y = 0`: the azimuth is drawn uniformly, and reflecting it reflects the
    /// direction. This is the invariant behind the `.cargo/mutants.toml` entry for the local
    /// estimator's `d[1] * view[1]` term — flipping that sign evaluates the phase function at
    /// `d · mirror(view)`, which is `mirror(d) · view`, and the ensemble contains `mirror(d)` with
    /// the same weight. The estimator's expectation is therefore unchanged and only its pairing of
    /// photons to values differs, which is Monte-Carlo noise rather than an answer.
    #[test]
    fn scattering_commutes_with_the_mirror_the_sun_plane_defines() {
        let mirror = |v: [f64; 3]| [v[0], -v[1], v[2]];
        for d in [
            [0.3, 0.5, -0.812_403_840_463_596],
            [0.6, -0.2, 0.775_886_584_517_1],
        ] {
            for cos_t in [-0.8_f64, 0.0, 0.35, 0.95] {
                for k in 0..8 {
                    let phi = TAU * f64::from(k) / 8.0;
                    let reflected = scatter_direction(mirror(d), cos_t, TAU - phi);
                    let direct = mirror(scatter_direction(d, cos_t, phi));
                    for i in 0..3 {
                        assert!(
                            (reflected[i] - direct[i]).abs() < 1e-15,
                            "cosθ={cos_t} φ={phi}: {reflected:?} against {direct:?}"
                        );
                    }
                }
            }
        }
    }

    /// (G1) Scattering rotates the direction, checked against an explicit rotation matrix.
    ///
    /// The slab only ever sees the vertical cosine, and a wrong rotation still produces a random
    /// walk whose reflectance and transmittance stay plausible — eleven mutants of this one line
    /// survived. Rebuild the geometry instead: a direction at polar angle `b` and azimuth `phi`
    /// from the z-axis is the vector `(sin b cos φ, sin b sin φ, cos b)`; carrying the old direction
    /// from the z-axis to polar angle `a` is `R_y(−a)`, whose third row is `(sin a, 0, cos a)`. The
    /// z-component of the product is what `scatter_mu` must return.
    #[test]
    fn scattering_rotates_the_direction_like_the_rotation_matrix_it_stands_for() {
        for ai in 1..12 {
            let a = PI * f64::from(ai) / 12.0;
            for bi in 1..12 {
                let b = PI * f64::from(bi) / 12.0;
                for pi in 0..12 {
                    let phi = TAU * f64::from(pi) / 12.0;
                    let v = (
                        libm::sin(b) * libm::cos(phi),
                        libm::sin(b) * libm::sin(phi),
                        libm::cos(b),
                    );
                    // R_y(−a) · v, third component; the first two only anchor the geometry.
                    let want = libm::sin(a) * v.0 + libm::cos(a) * v.2;
                    let got = scatter_mu(libm::cos(a), libm::cos(b), phi);
                    assert!(
                        (got - want).abs() < 1e-15,
                        "a={a} b={b} φ={phi}: {got} vs {want}"
                    );
                }
            }
        }
        // Straight up and straight down have no transverse component to rotate, whatever φ says.
        for phi in [0.0, 1.0, PI, 5.0] {
            assert_eq!(scatter_mu(1.0, 0.25, phi), 0.25);
            assert_eq!(scatter_mu(-1.0, 0.25, phi), -0.25);
        }
    }
}
