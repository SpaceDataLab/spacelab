//! Hexagonal ice crystals by geometric-optics ray tracing — the phase function with halos in it.
//!
//! Cirrus is the case a Henyey–Greenstein atmosphere fails on, and it fails on a *feature*: light
//! refracted through two alternate prism faces of a hexagonal column meets a 60° wedge and leaves at
//! a minimum deviation of 21.8°, and light entering a prism face and leaving a basal face meets a 90°
//! wedge and leaves at 45.7°. Those are the 22° and 46° halos. No value of `g` puts them in a smooth
//! one-lobe fit, and cirrus covers a large fraction of any optical Earth-observation scene, so the
//! error goes straight into a top-of-atmosphere correction.
//!
//! This traces rays through a randomly oriented hexagonal prism with Snell refraction and Fresnel
//! splitting, which is where those angles come from — they are not put in. The crystal is convex, so
//! the geometry is eight half-spaces and a slab intersection.
//!
//! # The two components, and how they are combined
//!
//! Geometric optics is the large-size-parameter limit, and it has **two** components:
//!
//! * **Ray tracing** — reflection and refraction through the crystal, from [`trace_on_grid`]. This is
//!   what carries the halos.
//! * **Fraunhofer diffraction** around the projected outline, from [`diffraction_on_grid`]. In the
//!   `x → ∞` limit it carries an equal share of the extinction — `Q_ext → 2`, half geometric and half
//!   diffracted — concentrated within roughly `1/x` radians of the forward direction.
//!
//! [`add_diffraction`] puts them together with the weights their scattering efficiencies imply,
//! `1` for diffraction and `ω_ray` for the traced part, and returns the combined optics. Two
//! consequences worth having in front of you: the combined single-scattering albedo is
//! `(1 + ω_ray)/2` — a *transparent* crystal is still `1`, but an absorbing one is pulled toward
//! `1/2` because diffraction cannot absorb — and the combined asymmetry parameter sits between the
//! ray-tracing `g` and the lobe's own, which is essentially 1. The halo angles are untouched: the
//! lobe is a few tenths of a degree wide at cirrus size parameters and the 22° peak is only rescaled.
//!
//! Ray tracing has a *lower* bound in validity: at size parameters below a few hundred the wave
//! nature of the field smooths the halo peaks and geometric optics overstates them. The crystals this
//! is for — tens to hundreds of microns at visible wavelengths — are comfortably above that, but the
//! model does not know it, so the caller must.
//!
//! # Roughness, and why real cirrus usually has no halo
//!
//! A perfect prism is the case that *maximises* the halo, and it is not the case the operational bulk
//! models are built from: Baum et al. (MODIS Collection 6) use severely roughened habits, which have
//! no halo at all. [`HexagonalCrystal::roughness`] is the same knob, `σ` — the standard deviation of
//! the micro-facet slope at each interaction, with the published labels `0` smooth, `0.03` moderately
//! roughened, `0.50` severely roughened (Yang et al. 2013).
//!
//! Measured here at `x = 200`, as the ratio of `p(22°)` to `p(21°)` — the sharp inner edge that makes
//! a halo visible:
//!
//! ```text
//!     σ = 0     31.3     σ = 0.03    1.86     σ = 0.15    1.03     σ = 0.50    0.90
//! ```
//!
//! So the feature is already most of the way gone at *moderate* roughening and completely gone at
//! severe, where the phase function decreases monotonically through the halo angles. That is the
//! result the published models encode, reproduced from the geometry rather than read out of a table.
//!
//! Gate: `derivations/ice_geometric_optics.mac` derives the Fresnel split and the refracted vector,
//! minimum deviation, Duff's frame, Cauchy's mean projected area, the `J₁` recurrence, Hankel
//! coefficients and series values, and the mass and moment identities of `reconstruct` and
//! `moments`; `derivations/ice_geometric_optics.gp` recomputes the numbers with PARI's `besselj`.
//! The tests `the_geometric_optics_reproduces_its_derivation` and
//! `the_reconstruction_keeps_every_interval_mass_and_the_moments_are_exact` tie the code to it.

// Ray tracing is written in the notation of optics (n, d, o, t, R), and the histogram bin counts
// become densities through an f64 divide. The Simpson panel counts are a real-valued span rounded
// down and then clamped into a positive range, so neither truncation nor sign loss can bite.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use crate::radiative_transfer::Rng;

/// Faces of the prism: six sides, then the two basal caps.
const PRISM_FACES: usize = 6;
/// Cap on internal reflections before a ray is written off as absorbed. Reached by a vanishing
/// fraction of rays; the count is what keeps a grazing path from looping.
const MAX_BOUNCES: u32 = 200;

/// A hexagonal ice column or plate of unit circumradius.
///
/// `aspect_ratio` is the *half-length* `L/2a`: above 1 a column, below 1 a plate. Both halo angles
/// come from wedge angles between faces and so are independent of it; the aspect ratio decides how
/// much energy reaches each wedge, and therefore the relative strength of the 22° and 46° peaks.
#[derive(Clone, Copy, Debug)]
pub struct HexagonalCrystal {
    /// Half-length divided by circumradius. Mid-latitude cirrus columns are near `2`.
    pub aspect_ratio: f64,
    /// Real refractive index. Ice at 550 nm is `1.3105`.
    pub refractive_index: f64,
    /// Imaginary refractive index (absorption). Ice at 550 nm is `≈ 3e-9`, i.e. transparent; at
    /// 2.2 µm it is `≈ 5e-4` and the crystal is not.
    pub absorption: f64,
    /// Size parameter `2πa/λ` of the circumradius. Enters only through absorption along the internal
    /// path — the ray-traced *angles* are scale-free, which is the defining approximation here.
    pub size_parameter: f64,
    /// Facet-tilt roughness `σ`: the standard deviation of the micro-facet **slope** at every
    /// reflection and refraction. `0` is a perfect prism; Yang et al. (2013) call `0.03` *moderately
    /// roughened* and `0.50` *severely roughened*, and it is the severely roughened habits that the
    /// operational bulk models are built from. Raising it washes the halos out, which is the whole
    /// reason a real cirrus sky usually has none.
    pub roughness: f64,
}

/// What a ray-tracing run measured about the crystal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayTracingOptics {
    /// Fraction of intercepted energy that left the crystal — the ray-tracing single-scattering
    /// albedo. `1` for non-absorbing ice.
    pub single_scattering_albedo: f64,
    /// Asymmetry parameter of the **ray-tracing component only**; see the module note on diffraction.
    pub asymmetry: f64,
    /// Rays that struck the crystal. The Monte-Carlo error on any bin scales as its inverse square
    /// root, so this is what a caller checks a halo peak's significance against.
    pub rays: u64,
}

/// Outward normal and plane offset of face `k`, for a crystal of unit circumradius.
fn face(k: usize, half_length: f64) -> ([f64; 3], f64) {
    match k.cmp(&PRISM_FACES) {
        core::cmp::Ordering::Less => {
            // A regular hexagon of circumradius 1 has inradius √3/2, and its face normals bisect the
            // vertices — hence the 30° offset. Alternate faces are 60° apart, the 22° wedge.
            let angle = core::f64::consts::PI / 6.0 + core::f64::consts::PI / 3.0 * k as f64;
            (
                [libm::cos(angle), libm::sin(angle), 0.0],
                libm::sqrt(3.0) / 2.0,
            )
        }
        core::cmp::Ordering::Equal => ([0.0, 0.0, 1.0], half_length),
        core::cmp::Ordering::Greater => ([0.0, 0.0, -1.0], half_length),
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Where a ray enters and leaves the convex crystal, with the faces it crosses.
///
/// The interior is the intersection of the eight half-spaces `n·x ≤ p`, so an entering face is one
/// the ray approaches (`n·d < 0`) and the entry is the *latest* of those; symmetrically the exit is
/// the earliest departing face. `None` when the ray misses.
fn intersect(
    origin: [f64; 3],
    direction: [f64; 3],
    half_length: f64,
) -> Option<(f64, usize, f64, usize)> {
    let (mut t_in, mut t_out) = (f64::NEG_INFINITY, f64::INFINITY);
    let (mut face_in, mut face_out) = (0usize, 0usize);
    for k in 0..PRISM_FACES + 2 {
        let (normal, offset) = face(k, half_length);
        let denominator = dot(normal, direction);
        let numerator = offset - dot(normal, origin);
        if denominator.abs() < 1e-15 {
            if numerator < 0.0 {
                return None; // parallel to the face and on its outside
            }
            continue;
        }
        let t = numerator / denominator;
        if denominator < 0.0 {
            if t > t_in {
                t_in = t;
                face_in = k;
            }
        } else if t < t_out {
            t_out = t;
            face_out = k;
        }
    }
    if t_in < t_out && t_out > 0.0 {
        Some((t_in, face_in, t_out, face_out))
    } else {
        None
    }
}

/// Unpolarised Fresnel reflectance, and the refracted direction when there is one.
///
/// `eta` is the ratio of the incident to the transmitted refractive index, `normal` points *out* of
/// the medium the ray is leaving, and `direction` is the incoming ray. Returns `None` for total
/// internal reflection, which is not a small reflectance — it is the absence of a transmitted ray.
fn refract(direction: [f64; 3], normal: [f64; 3], eta: f64) -> Option<(f64, [f64; 3])> {
    let cos_i = -dot(direction, normal);
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t >= 1.0 {
        return None;
    }
    let cos_t = libm::sqrt(1.0 - sin2_t);
    let r_s = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let r_p = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    let reflectance = 0.5 * (r_s * r_s + r_p * r_p);
    let factor = eta * cos_i - cos_t;
    let transmitted = [
        eta * direction[0] + factor * normal[0],
        eta * direction[1] + factor * normal[1],
        eta * direction[2] + factor * normal[2],
    ];
    Some((reflectance, transmitted))
}

/// Redraws before giving up on a micro-facet that is inconsistent with the face it sits on, and
/// falling back to the undistorted normal — which is always consistent, so no energy is ever lost to
/// a failed draw.
const ROUGHNESS_DRAWS: u32 = 32;

/// A micro-facet normal, tilted from the true one by a Gaussian slope of standard deviation `σ`.
///
/// The tilt is `(ζ_x, ζ_y)` drawn from an isotropic 2-D Gaussian in the facet's tangent plane, giving
/// the normal `n − ζ_x u − ζ_y v` renormalised — the Cai–Liou / Yang–Macke distorted-facet model, and
/// the same `σ` the published habits are labelled with. Only the *local* interaction sees it: the ray
/// still travels between the true faces, which is what keeps the crystal a closed convex body.
///
/// # Why a draw can be rejected
///
/// Tilt a facet far enough and its Fresnel outcome stops agreeing with the solid the face belongs to:
/// the reflected ray crosses the true face instead of bouncing off it, or the refracted ray turns
/// back the way it came. Left in, those rays walk out of the geometry and the tracer simply stops
/// following them — which is invisible as a wrong direction and very visible as a *transparent*
/// crystal whose single-scattering albedo falls to 0.77 at `σ = 0.5`. So a draw is accepted only if
/// the reflection stays on the ray's own side of the true face and the refraction crosses it, and is
/// otherwise redrawn. Clamping instead of redrawing would pile probability up at grazing incidence,
/// exactly where the Fresnel reflectance is most sensitive.
fn facet(
    direction: [f64; 3],
    facing: [f64; 3],
    outward: [f64; 3],
    eta: f64,
    sigma: f64,
    rng: &mut Rng,
) -> [f64; 3] {
    if sigma <= 0.0 {
        return facing;
    }
    // `−1` if the ray is outside the face, `+1` if inside. Reflection must keep that sign; refraction
    // must flip it.
    let side = if dot(direction, outward) < 0.0 {
        -1.0
    } else {
        1.0
    };
    let (u, v) = basis(facing);
    for _ in 0..ROUGHNESS_DRAWS {
        let radius = sigma * libm::sqrt(-2.0 * libm::log(rng.open_unit()));
        let phi = rng.azimuth();
        let (zx, zy) = (radius * libm::cos(phi), radius * libm::sin(phi));
        let tilted = [
            facing[0] - zx * u[0] - zy * v[0],
            facing[1] - zx * u[1] - zy * v[1],
            facing[2] - zx * u[2] - zy * v[2],
        ];
        let norm = libm::sqrt(dot(tilted, tilted));
        let unit = [tilted[0] / norm, tilted[1] / norm, tilted[2] / norm];
        if dot(direction, unit) >= 0.0 {
            continue;
        }
        if side * dot(reflect(direction, unit), outward) >= 0.0 {
            continue;
        }
        match refract(direction, unit, eta) {
            // Total internal reflection has no transmitted branch to check.
            None => return unit,
            Some((_, transmitted)) if side * dot(transmitted, outward) > 0.0 => return unit,
            Some(_) => {}
        }
    }
    facing
}

fn reflect(direction: [f64; 3], normal: [f64; 3]) -> [f64; 3] {
    let d = 2.0 * dot(direction, normal);
    [
        direction[0] - d * normal[0],
        direction[1] - d * normal[1],
        direction[2] - d * normal[2],
    ]
}

/// An orthonormal pair spanning the plane perpendicular to `w`, built without a branch on the
/// degenerate axis: Duff et al.'s branchless frame.
fn basis(w: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let sign = if w[2] >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + w[2]);
    let b = w[0] * w[1] * a;
    (
        [1.0 + sign * w[0] * w[0] * a, sign * b, -sign * w[0]],
        [b, sign + w[1] * w[1] * a, -w[1]],
    )
}

/// Trace `rays` through randomly oriented crystals and accumulate the scattering-angle distribution
/// onto the cosine grid `mu`, writing the **unnormalised** density into `out`.
///
/// Random orientation is imposed by drawing the incidence direction uniformly on the sphere in the
/// crystal's own frame, which is the same distribution as rotating the crystal and cheaper. The
/// impact point is drawn uniformly over a disc that encloses the crystal's projection, so rays are
/// weighted by projected area without that area ever being computed; misses are simply not counted.
///
/// Returns `None` if `mu` and `out` differ in length, if `mu` has fewer than two nodes, or if the
/// crystal parameters are not physical.
///
/// # Resolution
///
/// The halos are narrow — the 22° peak rises over about a degree — so the grid has to resolve them.
/// The counts land in the *interval* containing each cosine and are converted to node values by
/// averaging the two adjoining intervals, a two-interval boxcar that broadens a peak by one grid
/// spacing. At 0.25° spacing that is small against the peak's own width; at 2° it would erase it.
#[must_use]
// One ray's whole life — entry, Fresnel split, the internal bounce loop and the tally — is one
// story, and cutting it into pieces to satisfy a line count would hide where the weight goes.
#[allow(clippy::too_many_lines)]
pub fn trace_on_grid(
    crystal: &HexagonalCrystal,
    mu: &[f64],
    out: &mut [f64],
    rays: u64,
    seed: u64,
) -> Option<RayTracingOptics> {
    if mu.len() != out.len() || mu.len() < 2 {
        return None;
    }
    if !crystal.refractive_index.is_finite()
        || crystal.refractive_index <= 1.0
        || crystal.aspect_ratio <= 0.0
        || crystal.absorption < 0.0
        || crystal.size_parameter < 0.0
        || !crystal.roughness.is_finite()
        || crystal.roughness < 0.0
    {
        return None;
    }
    let half_length = crystal.aspect_ratio;
    let bound = libm::sqrt(1.0 + half_length * half_length) * 1.001;
    let extinction = 2.0 * crystal.absorption * crystal.size_parameter;

    let mut interval = alloc::vec![0.0_f64; mu.len() - 1];
    let mut rng = Rng::new(seed);
    let (mut hits, mut escaped, mut cosine_sum) = (0u64, 0.0_f64, 0.0_f64);

    let record = |cos_theta: f64, weight: f64, interval: &mut [f64]| {
        let clamped = cos_theta.clamp(-1.0, 1.0);
        let mut lo = 0usize;
        let mut hi = mu.len() - 1;
        while hi - lo > 1 {
            let mid = usize::midpoint(lo, hi);
            if mu[mid] <= clamped {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        interval[lo] += weight;
    };

    while hits < rays {
        // Uniform direction on the sphere, then a uniform impact point across the bounding disc.
        let z = 2.0 * rng.unit() - 1.0;
        let r = libm::sqrt((1.0 - z * z).max(0.0));
        let phi = rng.azimuth();
        let incident = [r * libm::cos(phi), r * libm::sin(phi), z];
        let (u, v) = basis(incident);
        let radius = bound * libm::sqrt(rng.unit());
        let angle = rng.azimuth();
        let (cu, su) = (radius * libm::cos(angle), radius * libm::sin(angle));
        let origin = [
            cu * u[0] + su * v[0] - 2.0 * bound * incident[0],
            cu * u[1] + su * v[1] - 2.0 * bound * incident[1],
            cu * u[2] + su * v[2] - 2.0 * bound * incident[2],
        ];

        let Some((t_in, face_in, _, _)) = intersect(origin, incident, half_length) else {
            continue;
        };
        hits += 1;

        let (true_normal, _) = face(face_in, half_length);
        let normal = facet(
            incident,
            true_normal,
            true_normal,
            1.0 / crystal.refractive_index,
            crystal.roughness,
            &mut rng,
        );
        let Some((reflectance, refracted)) =
            refract(incident, normal, 1.0 / crystal.refractive_index)
        else {
            // Cannot happen entering a denser medium; treat as a specular bounce rather than assume.
            let out_direction = reflect(incident, normal);
            escaped += 1.0;
            cosine_sum += dot(out_direction, incident);
            record(dot(out_direction, incident), 1.0, &mut interval);
            continue;
        };
        if rng.unit() < reflectance {
            let out_direction = reflect(incident, normal);
            escaped += 1.0;
            cosine_sum += dot(out_direction, incident);
            record(dot(out_direction, incident), 1.0, &mut interval);
            continue;
        }

        let mut point = [
            origin[0] + t_in * incident[0],
            origin[1] + t_in * incident[1],
            origin[2] + t_in * incident[2],
        ];
        let mut direction = refracted;
        let mut weight = 1.0_f64;
        for _ in 0..MAX_BOUNCES {
            let Some((_, _, t_out, face_out)) = intersect(point, direction, half_length) else {
                break;
            };
            weight *= libm::exp(-extinction * t_out);
            point = [
                point[0] + t_out * direction[0],
                point[1] + t_out * direction[1],
                point[2] + t_out * direction[2],
            ];
            let (outward, _) = face(face_out, half_length);
            // `refract` wants the normal facing the incoming ray. On the way *out* the ray and the
            // face's outward normal point the same way, so the inward one is what it must be given —
            // pass `outward` here and both the Fresnel coefficients and the transmitted direction
            // come out with the wrong sign, which shows up as a crystal that traps 94% of its light.
            let inward = [-outward[0], -outward[1], -outward[2]];
            let normal = facet(
                direction,
                inward,
                outward,
                crystal.refractive_index,
                crystal.roughness,
                &mut rng,
            );
            match refract(direction, normal, crystal.refractive_index) {
                Some((reflectance, transmitted)) if rng.unit() >= reflectance => {
                    let norm = libm::sqrt(dot(transmitted, transmitted));
                    let exit = [
                        transmitted[0] / norm,
                        transmitted[1] / norm,
                        transmitted[2] / norm,
                    ];
                    let cos_theta = dot(exit, incident);
                    escaped += weight;
                    cosine_sum += weight * cos_theta;
                    record(cos_theta, weight, &mut interval);
                    break;
                }
                _ => direction = reflect(direction, normal),
            }
        }
    }

    for i in 0..interval.len() {
        interval[i] /= mu[i + 1] - mu[i];
    }
    reconstruct(mu, &interval, out);

    Some(RayTracingOptics {
        single_scattering_albedo: escaped / hits as f64,
        asymmetry: cosine_sum / escaped,
        rays: hits,
    })
}

/// `J₁(x)` for `x ≥ 0`, by Miller's downward recurrence below `x = 100` and the Hankel asymptotic
/// expansion above it.
///
/// The recurrence `J_{n−1} = (2n/x) J_n − J_{n+1}` is unstable upward and stable downward, so it is
/// started at an order well above `x` from an arbitrary seed and normalised at the end by the
/// identity `1 = J₀ + 2(J₂ + J₄ + …)`. That is deliberately *not* the usual route: the polynomial
/// fits tabulated in the handbooks are shorter and faster, and they are also a transcription of
/// thirty magic constants that nothing in a test can distinguish from a typo. This has none — every
/// number in it is the recurrence or the normalisation identity — and it is accurate to `f64`
/// rounding at any argument rather than to the fit's `1e-8`.
fn bessel_j1(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return 0.0;
    }
    // Below this the series is J₁ = x/2 − x³/16 + …, and the second term is already under rounding.
    if x < 1e-8 {
        return 0.5 * x;
    }
    // Above this the Hankel asymptotic expansion costs a fixed nine terms where the recurrence costs
    // `start` steps, and `start` grows with `x` without bound — `diffraction_on_grid` takes the size
    // parameter from its caller, so that was an unbounded cost behind a public argument. Measured
    // against `mpmath.besselj` at 40 digits, 300 random arguments per band, as a fraction of the
    // envelope `√(2/πx)` — which is the honest measure, because near a zero of J₁ no method keeps
    // *relative* accuracy:
    //
    //     x ∈ [20, 40]     5.1e-11 asymptotic    1.7e-15 recurrence
    //     x ∈ [40, 80]     7.1e-14               2.3e-15
    //     x ∈ [80, 160]    8.1e-15               3.8e-15
    //     x ∈ [160, 400]   2.0e-14               9.4e-15
    //     x ∈ [400, 1e3]   2.0e-14               7.6e-15
    //     x ∈ [1e3, 1e4]   2.1e-13               3.6e-14
    //
    // so the crossover is where the two meet. Past 1e3 the asymptotic's floor is the argument
    // reduction in `cos(x − 3π/4)`, which costs `x·ε` and which the recurrence does not pay because it
    // never calls a trigonometric function; below 100 it is the expansion that is short of terms.
    if x >= 100.0 {
        // Abramowitz & Stegun 9.2.5–9.2.6 with `µ = 4ν² = 4`: `J₁ = √(2/πx)·(P cos χ − Q sin χ)` at
        // `χ = x − 3π/4`, with `aₖ = aₖ₋₁·(µ − (2k−1)²)/(8kx)`, `P` the even terms and `Q` the odd,
        // signed by `(−1)^⌊k/2⌋`. Every number in it is that recurrence — no tabulated fit, for the
        // same reason the downward recurrence below is not one.
        let (mut term, mut p, mut q) = (1.0_f64, 1.0_f64, 0.0_f64);
        for k in 1..=8_usize {
            let order = 2.0 * k as f64 - 1.0;
            term *= (4.0 - order * order) / (8.0 * k as f64 * x);
            let signed = if (k / 2) % 2 == 0 { term } else { -term };
            if k % 2 == 0 {
                p += signed;
            } else {
                q += signed;
            }
        }
        let chi = x - 0.75 * core::f64::consts::PI;
        return libm::sqrt(2.0 / (core::f64::consts::PI * x))
            * (p * libm::cos(chi) - q * libm::sin(chi));
    }
    // Start above the argument by a margin that grows as √x — the recurrence forgets its seed over
    // the region where J_n is exponentially small in n, and that region's width scales that way.
    let start = 2 * (((x + 10.0 * libm::sqrt(x) + 40.0) * 0.5) as usize + 1);
    let (mut current, mut next) = (1.0_f64, 0.0_f64);
    let (mut j1, mut normalisation) = (0.0_f64, 0.0_f64);
    for n in (1..=start).rev() {
        let previous = 2.0 * n as f64 / x * current - next;
        next = current;
        current = previous;
        // Even orders carry the normalisation sum; `current` is now J_{n−1}, so n odd means even.
        if n % 2 == 1 {
            normalisation += 2.0 * current;
        }
        if n == 2 {
            j1 = current;
        }
        if current.abs() > 1e250 {
            let scale = 1e-250;
            current *= scale;
            next *= scale;
            j1 *= scale;
            normalisation *= scale;
        }
    }
    // `current` is J₀ and the sum double-counted it, since n = 1 put it in.
    j1 / (normalisation - current)
}

/// Node values of the piecewise-linear density whose interval means are `interval`.
///
/// Each interval carries mass `mean × width`, which is split evenly between its two endpoints —
/// evenly because that is what puts the pair's centre of mass back at the interval's own. A node
/// then holds the mass of its two half-intervals and the trapezoid rule assigns it the weight
/// `(hᵢ₋₁ + hᵢ)/2`, so its value is the **width-weighted** mean of the two intervals meeting there.
///
/// The plain unweighted mean is what this replaces, and on a uniform grid the two agree. On a cosine
/// grid they do not: near `μ = 1` consecutive cells differ threefold in width, and a density with a
/// sharp forward peak had its zeroth moment inflated and its asymmetry parameter pushed 7% high — the
/// traced `g` came out `0.673` from a table whose rays averaged `0.628`. This form reproduces the
/// interval masses exactly and the first moment to second order in the spacing.
fn reconstruct(mu: &[f64], interval: &[f64], out: &mut [f64]) {
    let last = out.len() - 1;
    out[0] = interval[0];
    out[last] = interval[last - 1];
    for i in 1..last {
        let (before, after) = (mu[i] - mu[i - 1], mu[i + 1] - mu[i]);
        out[i] = (interval[i - 1] * before + interval[i] * after) / (before + after);
    }
}

/// The complete geometric-optics optical properties of a crystal: ray tracing plus diffraction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeometricOptics {
    /// Combined single-scattering albedo, `(1 + ω_ray)/2`. Diffraction scatters everything it
    /// intercepts, so it pulls an absorbing crystal's albedo toward `1/2` and leaves a transparent
    /// one at `1`.
    pub single_scattering_albedo: f64,
    /// Combined asymmetry parameter, the efficiency-weighted mean of the two components'.
    pub asymmetry: f64,
    /// Asymmetry of the ray-traced component alone.
    pub ray_asymmetry: f64,
    /// Asymmetry of the diffraction lobe alone — essentially 1 at any cirrus size parameter.
    pub diffraction_asymmetry: f64,
    /// Size parameter of the equal-area circular aperture the lobe is computed from.
    pub aperture_size_parameter: f64,
}

/// The size parameter of the circular aperture with the crystal's **mean projected area**.
///
/// Cauchy's theorem gives the orientation-averaged projected area of any convex body as a quarter of
/// its surface area, which for a hexagonal prism of unit circumradius and half-length `aspect` is
/// `(12·aspect + 3√3)/4`. Using that rather than a cross-section through the waist is what makes the
/// lobe's width right for a randomly oriented population instead of for one particular view.
fn aperture_size_parameter(crystal: &HexagonalCrystal) -> f64 {
    let surface = 12.0 * crystal.aspect_ratio + 3.0 * libm::sqrt(3.0);
    let projected = 0.25 * surface;
    crystal.size_parameter * libm::sqrt(projected / core::f64::consts::PI)
}

/// The largest number of Simpson panels [`diffraction_on_grid`] will lay across one grid interval.
///
/// The panel count is chosen so that consecutive samples are a quarter radian apart in
/// `v = x_eq θ`; the cap therefore turns into a domain on the product of the crystal and the grid:
/// an interval of width `Δθ` is resolvable only while `x_eq Δθ < 0.25 · MAX_LOBE_PANELS = 4096`.
/// Past that the pattern oscillates faster than the samples can follow, and Simpson aliases its
/// ringing into the tail — so the lobe refuses rather than return a plausible wrong shape.
///
/// The ceiling is far outside the physical range on any grid meant to carry a diffraction lobe:
/// the largest atmospheric ice crystal is about 1 cm, `x ≈ 5.7e4` at 550 nm and `x_eq ≈ 8.7e4`,
/// against 2.3e5 for the one-degree grid this module's tests use. It bites where it should — on a
/// grid too coarse for the crystal it was handed.
pub const MAX_LOBE_PANELS: usize = 16384;

/// The Fraunhofer diffraction lobe of the crystal's mean projected outline, onto the cosine grid
/// `mu`, written unnormalised into `out`.
///
/// The pattern of a circular aperture is `[2 J₁(v)/v]²` with `v = x_eq sin θ`. Two things are done
/// deliberately here rather than by point sampling:
///
/// * Each grid **interval** gets the *average* of the lobe over it, by Simpson with a sub-node count
///   set from how much of `v` the interval spans. At cirrus size parameters the lobe is narrower than
///   a quarter-degree grid cell, and a point sample would put its height at the mercy of where the
///   nodes happened to fall while losing most of its energy. The average keeps the integral right at
///   any resolution: an unresolved lobe becomes a correctly-weighted spike in the first interval.
/// * The `v → 0` limit is taken as `1` rather than evaluated.
///
/// The result is left **unnormalised**, and deliberately so: node values reconstructed from interval
/// averages do not integrate, under the consumer's own trapezoid rule, to the analytic integral of the
/// lobe — near `μ = 1` the cosine grid's intervals differ threefold in width and a sub-grid spike gets
/// redistributed across two of them. [`add_diffraction`] therefore normalises in exactly the metric
/// the consumer integrates with, which makes the energy split between the two components exact
/// whatever the reconstruction did to the shape inside the first cell.
///
/// Returns `None` if `mu` and `out` differ in length, if `mu` has fewer than two nodes, if the
/// crystal parameters are not physical, or if the grid cannot resolve the lobe — see
/// [`MAX_LOBE_PANELS`] — and otherwise the aperture size parameter it used.
#[must_use]
pub fn diffraction_on_grid(crystal: &HexagonalCrystal, mu: &[f64], out: &mut [f64]) -> Option<f64> {
    if mu.len() != out.len() || mu.len() < 2 || crystal.aspect_ratio <= 0.0 {
        return None;
    }
    let x_eq = aperture_size_parameter(crystal);
    if !x_eq.is_finite() || x_eq < 0.0 {
        return None;
    }
    // In the scattering angle, not its cosine. `v = x sin θ` has a bounded derivative in `θ` and an
    // unbounded one in `μ`: a node count chosen from the interval's `v`-span but laid out uniformly
    // in `μ` crowds to two samples per oscillation at the forward end of a wide cell, and Simpson
    // then aliases the ringing into a tail three orders of magnitude too bright.
    let integrand = |theta: f64| {
        let sine = libm::sin(theta);
        // `v = x θ`, not `x sin θ`. The two agree to `O(θ²)` across the whole lobe — its first zero
        // is at `θ = 3.83/x` — but `sin θ` vanishes again at `θ = π`, which would plant a second
        // aperture-bright lobe at exact backscatter carrying as much energy as the real one and drag
        // the asymmetry parameter to zero. The diffraction integral is a small-angle statement; this
        // is the form that says so.
        let v = x_eq * theta;
        if v < 1e-8 {
            sine
        } else {
            let t = 2.0 * bessel_j1(v) / v;
            t * t * sine
        }
    };

    let mut interval = alloc::vec![0.0_f64; mu.len() - 1];
    for i in 0..interval.len() {
        let (a, b) = (
            libm::acos(mu[i + 1].clamp(-1.0, 1.0)),
            libm::acos(mu[i].clamp(-1.0, 1.0)),
        );
        // `dv/dθ = x cos θ ≤ x`, so this bounds the `v`-step at a quarter radian — a dozen samples
        // across the π between consecutive zeros of the pattern.
        let demand = x_eq * (b - a) / 0.25;
        if demand >= MAX_LOBE_PANELS as f64 {
            return None;
        }
        let panels = (demand as usize + 1).max(2);
        let nodes = 2 * panels + 1;
        let h = (b - a) / (nodes - 1) as f64;
        let mut total = 0.0;
        for k in 0..nodes {
            let weight = if k == 0 || k == nodes - 1 {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            total += weight * integrand(a + h * k as f64);
        }
        // `∫ sin θ dθ` over the cell is exactly its width in `μ`, so this is the mean of the pattern.
        interval[i] = total * h / 3.0 / (mu[i + 1] - mu[i]);
    }

    reconstruct(mu, &interval, out);
    Some(x_eq)
}

/// `(∫p dμ, ∫μp dμ)` for the piecewise-linear interpolant of `value` on `mu`, both exact for it.
fn moments(mu: &[f64], value: &[f64]) -> (f64, f64) {
    let (mut zeroth, mut first) = (0.0, 0.0);
    for i in 0..mu.len() - 1 {
        let h = mu[i + 1] - mu[i];
        zeroth += 0.5 * h * (value[i] + value[i + 1]);
        first += h
            * (value[i] * (2.0 * mu[i] + mu[i + 1]) + value[i + 1] * (mu[i] + 2.0 * mu[i + 1]))
            / 6.0;
    }
    (zeroth, first)
}

/// Combine a ray-traced table with its diffraction lobe into the complete geometric-optics phase
/// function, in place.
///
/// `value` holds the unnormalised ray-traced density from [`trace_on_grid`] on entry and the combined
/// density on exit. The two components are weighted by the scattering efficiencies geometric optics
/// gives them — `1` for diffraction, `ω_ray` for the traced part, out of `Q_ext = 2` — so the result
/// is `(p_diff + ω_ray p_ray) / (1 + ω_ray)`.
///
/// Returns `None` on the same conditions as [`diffraction_on_grid`], or if the ray table integrates
/// to zero.
#[must_use]
pub fn add_diffraction(
    crystal: &HexagonalCrystal,
    rays: &RayTracingOptics,
    mu: &[f64],
    value: &mut [f64],
) -> Option<GeometricOptics> {
    let (ray_integral, ray_first) = moments(mu, value);
    if !ray_integral.is_finite() || ray_integral <= 0.0 {
        return None;
    }
    let mut lobe = alloc::vec![0.0_f64; mu.len()];
    let x_eq = diffraction_on_grid(crystal, mu, &mut lobe)?;
    let (lobe_integral, lobe_first) = moments(mu, &lobe);
    if !lobe_integral.is_finite() || lobe_integral <= 0.0 {
        return None;
    }

    let omega = rays.single_scattering_albedo;
    let total = 1.0 + omega;
    for (combined, diffracted) in value.iter_mut().zip(lobe.iter()) {
        *combined = (diffracted / lobe_integral + omega * *combined / ray_integral) / total;
    }
    let ray_asymmetry = ray_first / ray_integral;
    let diffraction_asymmetry = lobe_first / lobe_integral;
    Some(GeometricOptics {
        single_scattering_albedo: 0.5 * total,
        asymmetry: (diffraction_asymmetry + omega * ray_asymmetry) / total,
        ray_asymmetry,
        diffraction_asymmetry,
        aperture_size_parameter: x_eq,
    })
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::cast_lossless,
        clippy::unreadable_literal,
        clippy::excessive_precision
    )]

    use super::*;
    use crate::phase::{PhaseFunction, TabulatedPhase};
    use alloc::vec;
    use alloc::vec::Vec;

    const DEGREE: f64 = core::f64::consts::PI / 180.0;

    /// A quarter-degree grid in scattering angle, ascending in cosine.
    fn angle_grid() -> Vec<f64> {
        (0..721)
            .map(|i| libm::cos(core::f64::consts::PI * (720 - i) as f64 / 720.0).clamp(-1.0, 1.0))
            .collect()
    }

    fn cirrus() -> HexagonalCrystal {
        HexagonalCrystal {
            aspect_ratio: 2.0,
            refractive_index: 1.3105,
            absorption: 0.0,
            size_parameter: 200.0,
            roughness: 0.0,
        }
    }

    /// `p(θ)` of a normalised table at a scattering angle in degrees.
    fn at_degrees(mu: &[f64], value: &[f64], degrees: f64) -> f64 {
        let target = libm::cos(degrees * DEGREE);
        let index = mu.iter().position(|m| *m >= target).unwrap_or(mu.len() - 1);
        value[index]
    }

    fn traced() -> (Vec<f64>, Vec<f64>, RayTracingOptics) {
        let mu = angle_grid();
        let mut value = vec![0.0; mu.len()];
        let optics = trace_on_grid(&cirrus(), &mu, &mut value, 400_000, 7).unwrap();
        let mut cumulative = vec![0.0; mu.len()];
        TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).unwrap();
        (mu, value, optics)
    }

    /// (G3) The 22° halo stands where minimum deviation through a 60° ice wedge puts it, and no
    /// Henyey–Greenstein of the same asymmetry parameter has anything there.
    ///
    /// This is the discriminating measurement of the whole phase-function abstraction. Minimum
    /// deviation through a prism of apex `A` and index `n` is `2 asin(n sin(A/2)) − A`
    /// (`derivations/ice_geometric_optics.mac` shows it is the symmetric path's), which for
    /// `A = 60°` and `n = 1.3105` is **21.88°** — a number that comes from Snell's law and is never
    /// written into the code. Measured on the traced table: `p` rises from `0.241` at 21° to `7.52`
    /// at 22°, a factor of **31 across one degree**, then decays outward (7.41 at 23°, 4.45 at 25°,
    /// 1.59 at 30°) — the sharp inner edge and diffuse outer skirt that make a halo a halo.
    ///
    /// Henyey–Greenstein at the traced `g = 0.628` changes by 1% across the same degree, because it
    /// is smooth by construction. That is the failure the ticket names, stated as a ratio.
    #[test]
    fn the_twenty_two_degree_halo_stands_at_minimum_deviation_and_henyey_greenstein_has_none() {
        let (mu, value, optics) = traced();
        let apex = 60.0 * DEGREE;
        let minimum_deviation =
            2.0 * libm::asin(cirrus().refractive_index * libm::sin(0.5 * apex)) - apex;
        assert!(
            (minimum_deviation / DEGREE - 21.87721799402075648137179703994621629527).abs() < 1e-12,
            "minimum deviation {} deg",
            minimum_deviation / DEGREE
        );

        let (inside, peak) = (at_degrees(&mu, &value, 21.0), at_degrees(&mu, &value, 22.0));
        assert!(
            peak > 10.0 * inside,
            "no halo edge: p(21) = {inside}, p(22) = {peak}"
        );
        // ...and it is a peak, not a step: the outer skirt decays.
        assert!(at_degrees(&mu, &value, 30.0) < 0.5 * peak);

        let henyey = PhaseFunction::HenyeyGreenstein {
            asymmetry: optics.asymmetry,
        };
        let ratio = henyey.value(libm::cos(22.0 * DEGREE)) / henyey.value(libm::cos(21.0 * DEGREE));
        assert!(
            ratio < 1.1,
            "Henyey–Greenstein at g = {} produced a factor {ratio} across the halo edge",
            optics.asymmetry
        );
    }

    /// (G3) The 46° halo is there too, from the 90° wedge between a prism face and a basal face.
    ///
    /// Minimum deviation for `A = 90°` and `n = 1.3105` is 45.84°. It is a weaker feature than the
    /// 22° halo — measured `p` goes 0.274 at 45°, 0.695 at 46°, 0.717 at 47° — because a ray must
    /// find both a side and an end of the column, and that is a smaller share of the entry apertures.
    #[test]
    fn the_forty_six_degree_halo_comes_from_the_basal_prism_wedge() {
        let (mu, value, _) = traced();
        let apex = 90.0 * DEGREE;
        let minimum_deviation =
            2.0 * libm::asin(cirrus().refractive_index * libm::sin(0.5 * apex)) - apex;
        assert!(
            (minimum_deviation / DEGREE - 45.84107772585809094403802709225234137759).abs() < 1e-12,
            "minimum deviation {} deg",
            minimum_deviation / DEGREE
        );
        let (inside, peak) = (at_degrees(&mu, &value, 45.0), at_degrees(&mu, &value, 46.0));
        assert!(peak > 2.0 * inside, "no 46 deg edge: {inside} then {peak}");
    }

    /// Transparent ice scatters everything it intercepts, and the traced asymmetry is the
    /// ray-tracing one — forward-peaked, but nowhere near the `g → 1` that the missing diffraction
    /// lobe would push it to.
    #[test]
    fn a_transparent_crystal_scatters_everything_it_intercepts() {
        let (_, _, optics) = traced();
        assert!(
            optics.single_scattering_albedo > 0.9999,
            "omega = {}",
            optics.single_scattering_albedo
        );
        assert!(
            (0.5..0.8).contains(&optics.asymmetry),
            "ray-tracing g = {}",
            optics.asymmetry
        );
        assert_eq!(optics.rays, 400_000);
    }

    /// Absorption removes energy, monotonically in the absorption index — the one place the size
    /// parameter enters the ray trace at all.
    #[test]
    fn absorption_removes_energy_monotonically() {
        let mu = angle_grid();
        let mut value = vec![0.0; mu.len()];
        let mut previous = 1.1;
        for absorption in [0.0, 1e-5, 1e-4, 1e-3] {
            let crystal = HexagonalCrystal {
                absorption,
                ..cirrus()
            };
            let optics = trace_on_grid(&crystal, &mu, &mut value, 40_000, 3).unwrap();
            assert!(
                optics.single_scattering_albedo < previous,
                "k = {absorption}: omega {} did not fall below {previous}",
                optics.single_scattering_albedo
            );
            previous = optics.single_scattering_albedo;
        }
        assert!(previous < 0.8, "strong absorption left omega = {previous}");
    }

    /// (G3) Total internal reflection starts exactly at the critical angle, and normal-incidence
    /// reflectance is the textbook `((n−1)/(n+1))²`.
    ///
    /// Both are closed forms, and both are what the halo depends on: the critical angle decides
    /// which rays are trapped, and the Fresnel split decides how much of each wedge path survives.
    #[test]
    fn the_fresnel_interface_matches_its_closed_forms() {
        let n = 1.3105;
        let critical = libm::asin(1.0 / n);
        for (angle, expect_transmission) in [(critical - 1e-6, true), (critical + 1e-6, false)] {
            let direction = [libm::sin(angle), 0.0, -libm::cos(angle)];
            let outcome = refract(direction, [0.0, 0.0, -1.0], n);
            assert_eq!(
                outcome.is_some(),
                expect_transmission,
                "at {} deg from the critical angle",
                (angle - critical) / DEGREE
            );
        }
        let (reflectance, transmitted) =
            refract([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], 1.0 / n).unwrap();
        let want = ((n - 1.0) / (n + 1.0)) * ((n - 1.0) / (n + 1.0));
        assert!(
            (reflectance - want).abs() < 1e-15,
            "{reflectance} vs {want}"
        );
        // Normal incidence does not bend.
        assert!((transmitted[2] - 1.0).abs() < 1e-15, "{transmitted:?}");
    }

    /// The crystal is the intersection of eight half-spaces, and the geometry says so: a ray down
    /// the axis crosses the two basal faces at `±L/2`, a ray across the waist crosses two prism
    /// faces at `±√3/2`, and a ray offside misses.
    #[test]
    fn the_hexagonal_prism_is_the_solid_its_faces_describe() {
        let half_length = 2.0;
        let (t_in, face_in, t_out, face_out) =
            intersect([0.0, 0.0, -10.0], [0.0, 0.0, 1.0], half_length).unwrap();
        assert!((t_in - 8.0).abs() < 1e-12 && (t_out - 12.0).abs() < 1e-12);
        assert_eq!((face_in, face_out), (PRISM_FACES + 1, PRISM_FACES));

        // Across the waist the hexagon's reach depends on the azimuth, and both extremes are pinned:
        // the +x axis points at a *vertex*, so the boundary is the circumradius 1; the face normal at
        // 30° points at a face *centre*, so it is the inradius √3/2. A model that confused the two —
        // the easiest hexagon mistake there is — passes one of these and fails the other.
        let (t_in, face_in, t_out, _) =
            intersect([-10.0, 0.0, 0.0], [1.0, 0.0, 0.0], half_length).unwrap();
        assert!((t_in - 9.0).abs() < 1e-12, "vertex t_in {t_in}");
        assert!((t_out - 11.0).abs() < 1e-12, "vertex t_out {t_out}");
        assert!(face_in < PRISM_FACES);

        let inradius = libm::sqrt(3.0) / 2.0;
        let normal = [libm::cos(30.0 * DEGREE), libm::sin(30.0 * DEGREE), 0.0];
        let (t_in, _, t_out, _) = intersect(
            [-10.0 * normal[0], -10.0 * normal[1], 0.0],
            normal,
            half_length,
        )
        .unwrap();
        assert!((t_in - (10.0 - inradius)).abs() < 1e-12, "face t_in {t_in}");
        assert!(
            (t_out - (10.0 + inradius)).abs() < 1e-12,
            "face t_out {t_out}"
        );

        assert!(intersect([-10.0, 0.0, 3.0], [1.0, 0.0, 0.0], half_length).is_none());
        assert!(intersect([2.0, 0.0, -10.0], [0.0, 0.0, 1.0], half_length).is_none());
    }

    /// The parameters the model has no meaning for are refused rather than traced.
    #[test]
    fn unphysical_crystals_are_refused() {
        let mu = angle_grid();
        let mut value = vec![0.0; mu.len()];
        let mut short = [0.0; 1];
        assert!(trace_on_grid(&cirrus(), &[-1.0], &mut short, 10, 1).is_none());
        assert!(trace_on_grid(&cirrus(), &mu, &mut short, 10, 1).is_none());
        for bad in [
            HexagonalCrystal {
                refractive_index: 1.0,
                ..cirrus()
            },
            HexagonalCrystal {
                aspect_ratio: 0.0,
                ..cirrus()
            },
            HexagonalCrystal {
                absorption: -1.0,
                ..cirrus()
            },
            HexagonalCrystal {
                roughness: -0.1,
                ..cirrus()
            },
            HexagonalCrystal {
                roughness: f64::NAN,
                ..cirrus()
            },
        ] {
            assert!(trace_on_grid(&bad, &mu, &mut value, 10, 1).is_none());
        }
    }

    /// The face table, at the labels the rest of the module indexes it by.
    ///
    /// A regular hexagon of circumradius 1 has its face normals bisecting its vertices, so the `k`-th
    /// prism normal stands at `30° + 60°k` and its plane at the inradius `√3/2`. The *set* of six
    /// normals is symmetric under `k ↦ −k`, so nothing that only re-enters this table through the
    /// index it was handed can tell the ordering apart — which is why the ordering is asserted here
    /// rather than left to a solid-geometry check.
    #[test]
    fn the_face_normals_stand_where_the_hexagon_puts_them() {
        for k in 0..PRISM_FACES {
            let (normal, offset) = face(k, 2.0);
            let angle = (30.0 + 60.0 * k as f64) * DEGREE;
            assert!((normal[0] - libm::cos(angle)).abs() < 1e-15, "face {k} x");
            assert!((normal[1] - libm::sin(angle)).abs() < 1e-15, "face {k} y");
            assert!(normal[2] == 0.0, "face {k} z");
            assert!(
                (offset - libm::sqrt(3.0) / 2.0).abs() < 1e-15,
                "face {k} offset"
            );
        }
        assert_eq!(face(PRISM_FACES, 2.0), ([0.0, 0.0, 1.0], 2.0));
        assert_eq!(face(PRISM_FACES + 1, 2.0), ([0.0, 0.0, -1.0], 2.0));
    }

    /// (G1) Duff's branchless frame is orthonormal on both sides of the axis it is built to avoid.
    #[test]
    fn the_branchless_frame_is_orthonormal_on_both_sides_of_its_singular_axis() {
        for w in [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.0, 0.8],
            [0.36, 0.48, -0.8],
            [-0.48, 0.36, -0.8],
        ] {
            let (u, v) = basis(w);
            assert!((dot(u, u) - 1.0).abs() < 1e-14, "‖u‖ at {w:?}");
            assert!((dot(v, v) - 1.0).abs() < 1e-14, "‖v‖ at {w:?}");
            assert!(dot(u, v).abs() < 1e-14, "u·v at {w:?}");
            assert!(dot(u, w).abs() < 1e-14, "u·w at {w:?}");
            assert!(dot(v, w).abs() < 1e-14, "v·w at {w:?}");
        }
    }

    /// (G1) The unpolarised Fresnel reflectance away from normal incidence.
    ///
    /// At normal incidence `cos θₜ = 1` and the `p` branch's denominator is insensitive to how `η`
    /// and `cos θₜ` are combined, so a case at `0°` says nothing about it. These run to 80°, and the
    /// reference is written in the two-index form `(n₁cosθₜ − n₂cosθᵢ)/(n₁cosθₜ + n₂cosθᵢ)` rather
    /// than in `η`, so the two do not share an algebraic path.
    #[test]
    fn the_fresnel_reflectance_is_the_two_index_closed_form_at_oblique_incidence() {
        for (n1, n2) in [(1.0, 1.3105), (1.3105, 1.0)] {
            let eta = n1 / n2;
            for degrees in [0.0, 15.0, 30.0, 45.0, 60.0, 80.0] {
                let theta = degrees * DEGREE;
                let (cos_i, sin_i) = (libm::cos(theta), libm::sin(theta));
                let sin_t = eta * sin_i;
                if sin_t >= 1.0 {
                    assert!(
                        refract([sin_i, 0.0, -cos_i], [0.0, 0.0, 1.0], eta).is_none(),
                        "{n1}→{n2} at {degrees}° is past the critical angle"
                    );
                    continue;
                }
                let cos_t = libm::sqrt(1.0 - sin_t * sin_t);
                let r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
                let r_p = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i);
                let want = 0.5 * (r_s * r_s + r_p * r_p);
                let (got, _) = refract([sin_i, 0.0, -cos_i], [0.0, 0.0, 1.0], eta).unwrap();
                assert!(
                    (got - want).abs() < 1e-14,
                    "{n1}→{n2} at {degrees}°: R {got} against {want}"
                );
            }
        }
    }

    /// Which face a ray enters and leaves by, on the configurations the slab arithmetic is decided at.
    ///
    /// Three of these are ties, and a tie is where the entry and exit searches choose. A ray down the
    /// `x` axis hits a *vertex* of the hexagon, so two faces share both `t_in` and `t_out`, and only
    /// the strictness of the comparisons says which is recorded. The last two are the degenerate
    /// plate — the one geometry in which tangency and a zero-length exit are exact rather than nearly
    /// so — and they pin the two conditions that decide whether a ray hit the crystal at all.
    #[test]
    fn the_slab_intersection_resolves_its_own_ties() {
        // Along +x through the vertex at (1, 0, 0): faces 2 and 3 tie on entry, 0 and 5 on exit, and
        // the first of each pair wins because both comparisons are strict.
        let (t_in, face_in, t_out, face_out) =
            intersect([-10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 2.0).unwrap();
        assert!((t_in - 9.0).abs() < 1e-12, "t_in {t_in}");
        assert!((t_out - 11.0).abs() < 1e-12, "t_out {t_out}");
        assert_eq!((face_in, face_out), (2, 0));

        // A denominator of exactly 1e-15 is *not* parallel: the cap it belongs to is crossed, and
        // here it is crossed inside the prism, so it is the exit face.
        let (_, _, t_out, face_out) =
            intersect([-10.0, 0.0, 2.0 - 1e-14], [1.0, 0.0, 1e-15], 2.0).unwrap();
        assert_eq!(face_out, PRISM_FACES, "the +z cap is the exit face");
        assert!(
            t_out < 11.0,
            "the cap is crossed before the prism wall: {t_out}"
        );

        // Exactly *on* a face and parallel to it is inside, not outside.
        assert!(intersect([-10.0, 0.0, 2.0], [1.0, 0.0, 0.0], 2.0).is_some());

        // The degenerate plate, entered and left at the same instant: a tangency is a miss.
        assert!(intersect([-10.0, 0.0, -10.0], [1.0, 0.0, 1.0], 0.0).is_none());
        // Sitting on the exit face already, pointing out: an exit at `t = 0` is a miss too.
        assert!(intersect([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1.0).is_none());
    }

    /// The distorted micro-facet is the model it is named for, and the side test decides which rays
    /// can have one at all.
    ///
    /// The first half redraws the accepted tilt from a second generator of the same seed and rebuilds
    /// `n − ζ_x u − ζ_y v` from it, so every component of the Cai–Liou construction is pinned rather
    /// than only its length. The second half is a ray travelling *along* the face: `dot(d, outward)`
    /// is exactly zero, the ray is on neither side of it, and the reflection consistency test can
    /// then never pass — so every draw is rejected and the undistorted normal comes back, which is
    /// the whole point of having a fallback.
    #[test]
    fn the_micro_facet_is_the_cai_liou_tilt_and_a_ray_along_the_face_gets_no_tilt_at_all() {
        let eta = 1.0 / 1.3105;
        let sigma = 0.02;
        // A basal cap and a prism wall. The cap is the case that says nothing about the third
        // component: `basis([0, 0, 1])` is the `x`–`y` pair, so `u[2]` and `v[2]` are both zero and
        // the tilt's `z` terms vanish whatever sign they are given. The wall's frame has neither.
        for facing in [[0.0, 0.0, 1.0], face(0, 2.0).0] {
            let (u, v) = basis(facing);
            let (cosine, sine) = (libm::sqrt(1.0 - 0.04), 0.2);
            let direction = [
                sine * u[0] - cosine * facing[0],
                sine * u[1] - cosine * facing[1],
                sine * u[2] - cosine * facing[2],
            ];

            let mut rng = Rng::new(4242);
            let got = facet(direction, facing, facing, eta, sigma, &mut rng);

            let mut reference = Rng::new(4242);
            let radius = sigma * libm::sqrt(-2.0 * libm::log(reference.open_unit()));
            let phi = reference.azimuth();
            let (zx, zy) = (radius * libm::cos(phi), radius * libm::sin(phi));
            let tilted = [
                facing[0] - zx * u[0] - zy * v[0],
                facing[1] - zx * u[1] - zy * v[1],
                facing[2] - zx * u[2] - zy * v[2],
            ];
            let norm = libm::sqrt(dot(tilted, tilted));
            assert!(radius > 0.0, "the reference draw has to be a real tilt");
            for k in 0..3 {
                assert!(
                    (got[k] - tilted[k] / norm).abs() < 1e-15,
                    "facing {facing:?} component {k}: {} against {}",
                    got[k],
                    tilted[k] / norm
                );
            }
        }

        // Along the face, where `dot(d, outward)` is *exactly* zero: the ray is on neither side of
        // it, so the side factor takes its `+1` branch, the reflection-consistency test can never
        // pass, and every one of the 32 draws is rejected — which is what the undistorted fallback is
        // for. Only the cap is used here: the frame of a prism normal is orthogonal to a rounding
        // error rather than exactly, and a dot product of −1e-17 is a side.
        let facing = [0.0, 0.0, 1.0];
        let (u, _) = basis(facing);
        assert_eq!(dot(u, facing), 0.0);
        let mut rng = Rng::new(4242);
        assert_eq!(facet(u, facing, facing, eta, sigma, &mut rng), facing);
    }

    /// The tracer's own optics, reproduced.
    ///
    /// A Monte-Carlo run is deterministic given its seed, so these are numbers rather than bands, and
    /// what they pin is everything the loop does *before* the physics: the radius of the launch disc,
    /// the impact point on it, the offset that puts the origin behind the crystal, the refractive
    /// index handed to the micro-facet, the exponential along the internal path, and the
    /// normalisation of the exit direction. None of those move a bounded statistic like the albedo of
    /// a transparent crystal far enough to fail a band, and all of them move these.
    ///
    /// A number here changing means either the physics or the order the deviates are drawn in has
    /// changed. The second is not a defect, but it is not free either — every stream in the crate
    /// moves with it — so it should be a deliberate line in a diff rather than a surprise.
    #[test]
    fn the_ray_tracer_reproduces_its_own_measured_optics() {
        // aspect, absorption, size parameter, roughness, seed, albedo, asymmetry
        const CASES: [(f64, f64, f64, f64, u64, f64, f64); 5] = [
            (2.0, 0.0, 100.0, 0.0, 7, 1.0, 6.268_877_509_676_929e-1),
            (
                1.5,
                5e-4,
                300.0,
                0.0,
                11,
                5.467_196_166_251_665e-1,
                6.614_224_280_117_017e-1,
            ),
            (2.0, 0.0, 100.0, 0.3, 13, 1.0, 5.686_505_055_064_065e-1),
            (
                3.0,
                2e-4,
                150.0,
                0.05,
                17,
                8.637_092_940_593_799e-1,
                7.015_723_932_916_023e-1,
            ),
            (0.4, 0.0, 80.0, 0.0, 19, 1.0, 6.267_005_857_511_591e-1),
        ];
        let mu = angle_grid();
        for (aspect_ratio, absorption, size_parameter, roughness, seed, albedo, asymmetry) in CASES
        {
            let crystal = HexagonalCrystal {
                aspect_ratio,
                refractive_index: 1.3105,
                absorption,
                size_parameter,
                roughness,
            };
            let mut value = vec![0.0_f64; mu.len()];
            let optics = trace_on_grid(&crystal, &mu, &mut value, 20_000, seed).unwrap();
            assert_eq!(optics.rays, 20_000);
            assert!(
                (optics.single_scattering_albedo - albedo).abs() <= 1e-12 * albedo,
                "seed {seed}: ω {} against {albedo}",
                optics.single_scattering_albedo
            );
            assert!(
                (optics.asymmetry - asymmetry).abs() <= 1e-12 * asymmetry.abs(),
                "seed {seed}: g {} against {asymmetry}",
                optics.asymmetry
            );
            // `g` is a mean of cosines and cannot leave `[−1, 1]`; an exit direction that was not
            // renormalised would put it outside without changing anything else visibly.
            assert!(optics.asymmetry.abs() <= 1.0, "seed {seed}: g out of range");
            assert!((0.0..=1.0).contains(&optics.single_scattering_albedo));
        }
    }

    /// What the tracer accepts at the edges of its own domain.
    ///
    /// The ray-traced *angles* are scale-free — the size parameter enters only through absorption
    /// along the internal path — so a crystal with no size parameter is a legitimate request and not
    /// a degenerate one, and a negative size parameter is neither. Two nodes is a grid.
    #[test]
    fn the_tracer_accepts_a_scale_free_crystal_and_the_shortest_grid() {
        let crystal = |size_parameter| HexagonalCrystal {
            aspect_ratio: 2.0,
            refractive_index: 1.3105,
            absorption: 0.0,
            size_parameter,
            roughness: 0.0,
        };
        let mut two = [0.0_f64; 2];
        assert!(trace_on_grid(&crystal(50.0), &[-1.0, 1.0], &mut two, 200, 3).is_some());
        let mut out = [0.0_f64; 3];
        assert!(trace_on_grid(&crystal(0.0), &[-1.0, 0.0, 1.0], &mut out, 200, 3).is_some());
        assert!(trace_on_grid(&crystal(-1.0), &[-1.0, 0.0, 1.0], &mut out, 200, 3).is_none());
    }

    /// The domain guards of `bessel_j1`, and its series cutoff.
    ///
    /// The recurrence is meaningless at or below zero and at a non-finite argument, and below the
    /// cutoff the answer is the first term of the series and nothing else — the second is `x³/16`,
    /// which at the cutoff is 6e-26 of the first and so eight decades under rounding. Both are
    /// returned exactly, and asserting them exactly is what separates the cutoff from the recurrence.
    ///
    /// `1e-17` is in the list because the other two are not enough: the recurrence and the series
    /// happen to agree bit-for-bit at `1e-30` and `1e-12`, so an implementation that took the
    /// *recurrence* below the cutoff would pass on those two and only be wrong here, where the
    /// recurrence lands on `4.999_999_999_999_999_59e-18` against the series' `5e-18`.
    #[test]
    fn the_bessel_function_refuses_what_it_cannot_recur_and_uses_its_series_below_the_cutoff() {
        for bad in [
            0.0,
            -1e-30,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert!(bessel_j1(bad) == 0.0, "J₁({bad}) = {}", bessel_j1(bad));
        }
        assert_eq!(bessel_j1(1e-30), 0.5e-30);
        assert_eq!(bessel_j1(1e-17), 0.5e-17);
        assert_eq!(bessel_j1(1e-12), 0.5e-12);
    }

    /// The lobe's inner quadrature, reassembled panel by panel.
    ///
    /// `diffraction_on_grid` walks each grid interval once and picks every sub-node's weight from the
    /// `1-4-2-…-4-1` pattern; this applies the three-point rule to each pair of sub-intervals
    /// instead and lets the shared nodes accumulate. Same rule, assembled the other way round. It
    /// deliberately reproduces the *scale* as well as the shape: `out` is documented unnormalised, so
    /// its absolute size is not a physical quantity, but it is the accumulated quadrature weight, and
    /// comparing it is the only thing that makes a mutation which merely rescales every cell visible.
    fn panelwise_lobe(crystal: &HexagonalCrystal, mu: &[f64]) -> Vec<f64> {
        let x_eq = aperture_size_parameter(crystal);
        let pattern = |theta: f64| {
            let v = x_eq * theta;
            let sine = libm::sin(theta);
            if v < 1e-8 {
                sine
            } else {
                let t = 2.0 * bessel_j1(v) / v;
                t * t * sine
            }
        };
        let mut interval = vec![0.0_f64; mu.len() - 1];
        for i in 0..interval.len() {
            let (a, b) = (
                libm::acos(mu[i + 1].clamp(-1.0, 1.0)),
                libm::acos(mu[i].clamp(-1.0, 1.0)),
            );
            let panels = ((x_eq * (b - a) / 0.25) as usize + 1).clamp(2, 16384);
            let h = (b - a) / (2 * panels) as f64;
            let mut total = 0.0;
            for k in 0..panels {
                let start = a + h * (2 * k) as f64;
                total += h / 3.0
                    * (pattern(start) + 4.0 * pattern(start + h) + pattern(start + 2.0 * h));
            }
            interval[i] = total / (mu[i + 1] - mu[i]);
        }
        let mut out = vec![0.0_f64; mu.len()];
        reconstruct(mu, &interval, &mut out);
        out
    }

    /// The lobe is the interval average its quadrature claims to compute, on a resolved crystal and
    /// on one whose lobe is far narrower than a grid cell.
    #[test]
    fn the_lobe_is_the_interval_average_the_quadrature_claims_to_compute() {
        let mu = angle_grid();
        for size_parameter in [20.0, 500.0] {
            let crystal = HexagonalCrystal {
                aspect_ratio: 2.0,
                refractive_index: 1.3105,
                absorption: 0.0,
                size_parameter,
                roughness: 0.0,
            };
            let mut got = vec![0.0_f64; mu.len()];
            diffraction_on_grid(&crystal, &mu, &mut got).unwrap();
            let want = panelwise_lobe(&crystal, &mu);
            for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-10 * b.abs().max(1e-300),
                    "x = {size_parameter}, node {i}: {a} against the panelwise {b}"
                );
            }
        }
    }

    /// What the lobe accepts and what it refuses, at the edges of both.
    ///
    /// A crystal with no size parameter still has an outline — the geometric-optics angles are
    /// scale-free and `trace_on_grid` admits `x = 0` for that reason — so the lobe is defined there
    /// and is just the `v → 0` limit. A *negative* size parameter is not a small aperture, it is not
    /// an aperture, and two nodes is a grid.
    #[test]
    fn the_lobe_accepts_a_scale_free_crystal_and_the_shortest_grid_and_refuses_a_negative_aperture()
    {
        let crystal = |size_parameter| HexagonalCrystal {
            aspect_ratio: 2.0,
            refractive_index: 1.3105,
            absorption: 0.0,
            size_parameter,
            roughness: 0.0,
        };
        let mut two = [0.0_f64; 2];
        assert!(diffraction_on_grid(&crystal(50.0), &[-1.0, 1.0], &mut two).is_some());
        let mut out = [0.0_f64; 3];
        assert!(diffraction_on_grid(&crystal(0.0), &[-1.0, 0.0, 1.0], &mut out).is_some());
        assert!(diffraction_on_grid(&crystal(-1.0), &[-1.0, 0.0, 1.0], &mut out).is_none());
    }

    /// The lobe refuses the crystal-and-grid pairs its quadrature cannot resolve, and the ceiling is
    /// the grid's, not the crystal's.
    ///
    /// The panel count wants `x_eq Δθ / 0.25` samples per interval, so [`MAX_LOBE_PANELS`] is a
    /// statement about the *product*. On the two-node grid — one interval of width π — it becomes a
    /// ceiling on `x_eq` alone, and the same crystal that the two-node grid refuses is resolved
    /// without complaint on a quarter-degree one, as is a 1 cm crystal at 550 nm, `x ≈ 5.7e4`, which
    /// is the largest that occurs in the atmosphere.
    #[test]
    fn the_lobe_refuses_a_grid_that_cannot_resolve_it() {
        let crystal = |size_parameter| HexagonalCrystal {
            aspect_ratio: 2.0,
            refractive_index: 1.3105,
            absorption: 0.0,
            size_parameter,
            roughness: 0.0,
        };
        let ceiling = 0.25 * MAX_LOBE_PANELS as f64 / core::f64::consts::PI;
        let per_size = aperture_size_parameter(&crystal(1.0));
        let (under, over) = (0.999 * ceiling / per_size, 1.001 * ceiling / per_size);

        let mut two = [0.0_f64; 2];
        assert!(diffraction_on_grid(&crystal(under), &[-1.0, 1.0], &mut two).is_some());
        assert!(diffraction_on_grid(&crystal(over), &[-1.0, 1.0], &mut two).is_none());

        let mu = angle_grid();
        let mut out = vec![0.0_f64; mu.len()];
        for size_parameter in [over, 5.7e4] {
            assert!(
                diffraction_on_grid(&crystal(size_parameter), &mu, &mut out).is_some(),
                "x = {size_parameter} is inside the quarter-degree grid's domain"
            );
        }
    }

    /// The two component asymmetries `add_diffraction` reports are the moments of the two tables it
    /// combined, and the combined table is the weighted mean it says it is.
    ///
    /// Everything else about this function is checked through the energy split, which is a ratio and
    /// cancels every scale in it. These are the numbers themselves: each reported asymmetry against
    /// the first moment of its own table, and the combined density against the two it was built from,
    /// node by node.
    #[test]
    fn the_combined_density_and_both_asymmetries_are_the_moments_of_the_tables_they_came_from() {
        let mu = angle_grid();
        let crystal = HexagonalCrystal {
            aspect_ratio: 1.5,
            refractive_index: 1.3105,
            absorption: 5e-4,
            size_parameter: 120.0,
            roughness: 0.0,
        };
        let mut value = vec![0.0_f64; mu.len()];
        let rays = trace_on_grid(&crystal, &mu, &mut value, 20_000, 11).unwrap();
        let ray_table = value.clone();
        let lobe = panelwise_lobe(&crystal, &mu);

        let (ray_integral, ray_first) = moments(&mu, &ray_table);
        let (lobe_integral, lobe_first) = moments(&mu, &lobe);
        let optics = add_diffraction(&crystal, &rays, &mu, &mut value).unwrap();

        assert!(
            (optics.ray_asymmetry - ray_first / ray_integral).abs() < 1e-12,
            "ray g {} against the table's own first moment {}",
            optics.ray_asymmetry,
            ray_first / ray_integral
        );
        assert!(
            (optics.diffraction_asymmetry - lobe_first / lobe_integral).abs() < 1e-10,
            "lobe g {} against the lobe's own first moment {}",
            optics.diffraction_asymmetry,
            lobe_first / lobe_integral
        );

        let omega = rays.single_scattering_albedo;
        let total = 1.0 + omega;
        for i in 0..mu.len() {
            let want = (lobe[i] / lobe_integral + omega * ray_table[i] / ray_integral) / total;
            assert!(
                (value[i] - want).abs() <= 1e-9 * want.abs().max(1e-300),
                "node {i}: combined {} against {want}",
                value[i]
            );
        }
    }

    /// (G2) `J₁`, its first zero and the aperture size parameter are their derivation's numbers.
    ///
    /// `derivations/ice_geometric_optics.mac` sums the series `J₁ = Σ (−1)ᵏ (x/2)^{2k+1}/(k!(k+1)!)`
    /// in bigfloat, at a precision that carries its largest term, and finds the first zero on it; it
    /// also shows the three-term recurrence and the Hankel coefficients the code runs on are the
    /// series' and A&S 9.2.9–9.2.10's. `.gp` recomputes every value with PARI's own `besselj`. The
    /// arguments are the thirteen of the comparator test below. The bound is absolute, as there,
    /// because near a zero of `J₁` no method keeps relative accuracy. The aperture is
    /// `x √(S/4π)` with Cauchy's `S/4` for the mean projected area and `S = 12h + 3√3`.
    #[test]
    fn the_geometric_optics_reproduces_its_derivation() {
        const J1: [(f64, f64); 13] = [
            (1e-6, 4.999999999999375000000000026041666666666e-7),
            (2.5e-5, 1.249999999902343750002543131510383552975e-5),
            (0.5, 2.422684576748738863839545761415316408006e-1),
            (1.0, 4.400505857449335159596822037189149131274e-1),
            (2.0, 5.767248077568733872024482422691370869203e-1),
            (5.0, -3.275791375914652220377343219101691327609e-1),
            (8.0, 2.346363468539146243812766515904546115488e-1),
            (10.0, 4.347274616886143666974876802585928830627e-2),
            (20.0, 6.6833124175850045578992974193646719983e-2),
            (50.0, -9.751182812517513766145895387370161404049e-2),
            (100.0, -7.714535201411215803268549492723447021161e-2),
            (400.0, -9.222058428586351254235216748846469409394e-3),
            (1000.0, 4.728311907089523917576071901216916285418e-3),
        ];
        const FIRST_ZERO: f64 = 3.831705970207512315614435886308160766565;
        const APERTURE: f64 = 3.048511760626216161463692935850979915572e2;
        // Measured: at most 1.2e-16 below x = 100, and 7.9e-16 at x = 400 on the asymptotic branch,
        // whose floor is the argument reduction in `cos(x − 3π/4)`.
        for (x, want) in J1 {
            let got = bessel_j1(x);
            assert!(
                (got - want).abs() < 1e-15,
                "J1({x}) = {got:e} against {want:e}"
            );
        }
        // The double nearest the zero is within half an ulp, 2.2e-16, of it, and |J₁′| there is 0.40.
        assert!(bessel_j1(FIRST_ZERO).abs() < 2e-16);
        let x_eq = aperture_size_parameter(&cirrus());
        assert!(
            (x_eq - APERTURE).abs() <= 4.0 * f64::EPSILON * APERTURE,
            "x_eq {x_eq} against {APERTURE}"
        );
    }

    /// (G4) `reconstruct` keeps every interval's mass under the trapezoid rule, and `moments` is
    /// exact on the piecewise-linear interpolant.
    ///
    /// `derivations/ice_geometric_optics.mac` shows both for general spacings: a node holding the
    /// width-weighted mean of its two intervals, weighted by the trapezoid's `(hᵢ₋₁ + hᵢ)/2`, puts
    /// back exactly `Σ Iᵢ hᵢ`, and the two segment formulas are the integrals of the linear
    /// interpolant. A linear density `a + bμ` is its own interpolant, so its moments are `2a` and
    /// `2b/3` on any grid.
    #[test]
    fn the_reconstruction_keeps_every_interval_mass_and_the_moments_are_exact() {
        let mu = angle_grid();
        let interval: Vec<f64> = (0..mu.len() - 1)
            .map(|i| 1.0 + libm::sin(0.37 * i as f64) * libm::sin(0.37 * i as f64))
            .collect();
        let mut node = vec![0.0_f64; mu.len()];
        reconstruct(&mu, &interval, &mut node);
        let want: f64 = (0..interval.len())
            .map(|i| interval[i] * (mu[i + 1] - mu[i]))
            .sum();
        let (got, _) = moments(&mu, &node);
        assert!(
            (got - want).abs() <= 1e-14 * want,
            "trapezoid mass {got} against the interval mass {want}"
        );

        let (a, b) = (0.3, 1.4);
        let linear: Vec<f64> = mu.iter().map(|m| a + b * m).collect();
        let (zeroth, first) = moments(&mu, &linear);
        assert!((zeroth - 2.0 * a).abs() < 1e-14, "zeroth {zeroth}");
        assert!((first - 2.0 * b / 3.0).abs() < 1e-14, "first {first}");
    }

    /// `J₁` against an independent implementation, and against its own defining zero.
    ///
    /// The reference values are `scipy.special.j1` at arguments spanning nine decades — `2.5e-5` is
    /// `mpmath.besselj` at 40 digits, for the reason below — chosen to straddle the places a Bessel
    /// routine goes wrong: the small-argument limit where `J₁ ≈ x/2`,
    /// the first maximum, the first zero, and the large-argument regime where the recurrence has to
    /// be started far above `x` and normalised. The zero at `3.831705970207512…` is the stronger of
    /// the two checks, because it is a property of the function that no implementation can fit its
    /// way to: it is tested as an absolute residual, which is the only sensible tolerance at a root.
    ///
    /// `2.5e-5` is there for a reason none of the others cover. The unnormalised recurrence overflows
    /// and is rescaled by `1e-250` whenever it passes `1e250`, and for `x` in `[2.10e-5, 3.67e-5]` —
    /// and, over a 5 201-point log sweep of `[1e-8, 1e5]`, only there — that rescale fires at `n = 2`,
    /// after `j1` has already been assigned. Everywhere else it fires at `n ≥ 3`, where `j1` is still
    /// zero and any scaling of it is a no-op that the assignment then overwrites. So this is the one
    /// argument at which the rescale's effect on `j1` is observable at all, and without it a rescale
    /// that added or divided instead of multiplying would return `1e245` or `∞` unremarked.
    #[test]
    fn the_bessel_function_matches_an_independent_implementation() {
        const REFERENCE: [(f64, f64); 13] = [
            (1e-6, 4.999_999_999_999_375e-7),
            (2.5e-5, 1.249_999_999_902_343_8e-5),
            (0.5, 0.242_268_457_674_873_87),
            (1.0, 0.440_050_585_744_933_55),
            (2.0, 0.576_724_807_756_873_4),
            (5.0, -0.327_579_137_591_465_3),
            (8.0, 0.234_636_346_853_914_6),
            (10.0, 0.043_472_746_168_861_445),
            (20.0, 0.066_833_124_175_850_04),
            (50.0, -0.097_511_828_125_175_17),
            (100.0, -0.077_145_352_014_112_17),
            (400.0, -0.009_222_058_428_586_357),
            (1000.0, 0.004_728_311_907_089_521),
        ];
        for (x, expected) in REFERENCE {
            let residual = (bessel_j1(x) - expected).abs();
            assert!(
                residual < 2e-15,
                "J1({x}) = {} against {expected}, residual {residual:e}",
                bessel_j1(x)
            );
        }
        // The first zero of J₁, to all the digits a f64 argument can carry.
        assert!(bessel_j1(3.831_705_970_207_512_3).abs() < 2e-15);
        assert_eq!(bessel_j1(0.0), 0.0);
    }

    /// `x = 100` is a seam between two entirely different algorithms, and a seam is where a wrong
    /// constant hides: both branches can be individually accurate and the pair still step.
    ///
    /// Measured across the two adjacent doubles either side of it, nothing else can contribute:
    /// `|J₁′| ≤ 0.08` over a gap of 1.4e-14 is 1.1e-15, the same size as the asymptotic's own 6.5e-16
    /// absolute accuracy there, so the bound below is the two branches' disagreement and nothing more.
    /// The seam cannot drift *downward* unnoticed either — the `x = 50` row of the table above holds
    /// `J₁` to 2e-15 absolute, and the expansion is 8.0e-15 out there.
    #[test]
    fn the_two_branches_of_the_bessel_function_meet_at_their_crossover() {
        let seam = 100.0_f64;
        let under = f64::from_bits(seam.to_bits() - 1);
        let step = (bessel_j1(seam) - bessel_j1(under)).abs();
        assert!(
            step < 3e-15,
            "J₁ steps by {step:e} across the crossover: {} then {}",
            bessel_j1(under),
            bessel_j1(seam)
        );
    }

    /// (G3) The lobe is the Airy pattern of the crystal's **mean projected area**, and its first zero
    /// says so.
    ///
    /// Cauchy's theorem puts the orientation-averaged projected area of a convex body at a quarter of
    /// its surface. For the aspect-2 prism of unit circumradius that is `(12·2 + 3√3)/4 = 7.29904`,
    /// an equal-area disc of radius `1.524256`, and at `x = 200` an aperture size parameter of
    /// `304.851`. The Airy pattern's first zero sits at `v = 3.8317`, and the lobe is written in
    /// `v = x θ`, so the dark ring must land at `3.8317/304.851 rad = 0.72016°`. Nothing in the code
    /// contains that angle; it follows from the crystal's geometry and the zero of `J₁`, and it is
    /// the one measurement that would catch a lobe built from the wrong cross-section — a waist cut
    /// rather than the orientation average would misplace it by a factor of order one.
    #[test]
    fn the_diffraction_lobe_is_the_airy_pattern_of_the_mean_projected_area() {
        let crystal = cirrus();
        let x_eq = aperture_size_parameter(&crystal);
        assert!((x_eq - 304.851_176_062_621_6).abs() < 1e-9);

        // A grid fine enough to resolve a lobe 0.72° wide: 0.005° out to 3°.
        let fine: Vec<f64> = (0..=600)
            .map(|i| libm::cos((600 - i) as f64 * 0.005 * DEGREE).clamp(-1.0, 1.0))
            .collect();
        let mut lobe = vec![0.0; fine.len()];
        assert_eq!(diffraction_on_grid(&crystal, &fine, &mut lobe), Some(x_eq));

        let peak = lobe[fine.len() - 1];
        let trough = (0..fine.len() - 1)
            .rfind(|i| lobe[*i] < lobe[i + 1] && (*i == 0 || lobe[*i] < lobe[i - 1]))
            .expect("the pattern has a dark ring");
        let angle = libm::acos(fine[trough]) / DEGREE;
        let expected = 3.831_705_970_207_512_3 / x_eq / DEGREE;
        assert!(
            (angle - expected).abs() < 0.01,
            "first zero at {angle}° against {expected}°"
        );
        assert!(
            lobe[trough] < peak * 1e-3,
            "the ring is dark: {} against a peak of {peak}",
            lobe[trough]
        );
    }

    /// Diffraction cannot absorb, so it pulls the combined albedo halfway to 1 — and leaves a
    /// transparent crystal exactly where it was.
    ///
    /// This is the whole content of the `Q_ext → 2` split stated as an assertion. A crystal that
    /// swallows 40% of the light it refracts still returns three fifths of what it intercepts,
    /// because half of that was never inside it.
    #[test]
    fn diffraction_lifts_an_absorbing_crystals_albedo_halfway_to_one() {
        let mu = angle_grid();
        for (absorption, expected_ray) in [(0.0, 1.0), (0.05, 0.0), (0.5, 0.0)] {
            let crystal = HexagonalCrystal {
                absorption,
                ..cirrus()
            };
            let mut value = vec![0.0; mu.len()];
            let rays = trace_on_grid(&crystal, &mu, &mut value, 40_000, 11).unwrap();
            let optics = add_diffraction(&crystal, &rays, &mu, &mut value).unwrap();
            if absorption == 0.0 {
                assert!((rays.single_scattering_albedo - expected_ray).abs() < 1e-4);
            } else {
                assert!(rays.single_scattering_albedo < 1.0);
            }
            assert!(
                (optics.single_scattering_albedo - 0.5 * (1.0 + rays.single_scattering_albedo))
                    .abs()
                    < 1e-15
            );
            assert!(optics.single_scattering_albedo >= rays.single_scattering_albedo - 1e-12);
            assert!(optics.single_scattering_albedo >= 0.5);
        }
    }

    /// The combined table is a proper density whose two components carry the energy geometric optics
    /// gives them: `1` for diffraction and `ω_ray` for the traced part.
    ///
    /// Three things are checked against each other rather than against a constant — the integral is
    /// 1, the first moment is the reported asymmetry, and the halo at 22° is the traced halo scaled
    /// by exactly `ω/(1+ω)`. The last is what would break if the weights were swapped or the lobe
    /// left unnormalised, and it is checked at 22° precisely because the lobe has nothing there.
    #[test]
    fn the_combined_table_splits_the_energy_by_scattering_efficiency() {
        let (mu, ray, rays) = traced();
        let mut value = ray.clone();
        let optics = add_diffraction(&cirrus(), &rays, &mu, &mut value).unwrap();

        let (integral, first) = moments(&mu, &value);
        assert!((integral - 1.0).abs() < 1e-12, "integral {integral}");
        assert!((first / integral - optics.asymmetry).abs() < 1e-12);

        // The lobe only ever adds, and how much it adds is the Airy tail — which is a real part of
        // the model, not an error term. Measured against the traced value at `x = 200`: `+449%` at
        // 5°, `+99%` at 10°, `+28%` at 15°, then `+0.41%` at 22°, `+0.64%` at 46° and `+0.24%` at
        // 120°. That profile is the reason the halo angles survive being averaged with something
        // that carries half the energy.
        let share = rays.single_scattering_albedo / (1.0 + rays.single_scattering_albedo);
        for (angle, floor, ceiling) in [
            (5.0, 1.0, f64::INFINITY),
            (10.0, 0.5, f64::INFINITY),
            (22.0, 0.0, 0.02),
            (46.0, 0.0, 0.02),
            (120.0, 0.0, 0.02),
        ] {
            let excess =
                at_degrees(&mu, &value, angle) / (at_degrees(&mu, &ray, angle) * share) - 1.0;
            assert!(
                excess >= floor && excess <= ceiling,
                "at {angle}° the lobe adds {excess}, outside [{floor}, {ceiling}]"
            );
        }

        assert!(optics.diffraction_asymmetry > 0.95, "the lobe is forward");
        assert!(optics.asymmetry > optics.ray_asymmetry);
        assert!(optics.asymmetry < optics.diffraction_asymmetry);
        // The table's first moment against the ray tracer's own running mean of `cos Θ` — two
        // estimators of the same number that share no arithmetic. They agree to `6e-6`, which is what
        // says the node reconstruction is area- and centroid-preserving rather than merely plausible.
        assert!(
            (optics.ray_asymmetry - rays.asymmetry).abs() < 1e-4,
            "table {} against tracer {}",
            optics.ray_asymmetry,
            rays.asymmetry
        );
    }

    /// The lobe keeps its share of the energy on a grid far too coarse to resolve it.
    ///
    /// At `x = 200` the pattern is 0.72° across and a quarter-degree grid barely samples it; a
    /// one-degree grid does not sample it at all. Point-evaluating the Airy function at the nodes
    /// would put a number of order `1e-4` in the first cell and lose the lobe outright, and the loss
    /// would depend on where the nodes happened to fall. Integrating each interval instead makes the
    /// unresolved lobe a correctly-weighted spike, and the test is that the combined asymmetry
    /// parameter — which is the quantity a radiative-transfer solver actually consumes — moves by
    /// under `1e-3` when the grid is coarsened fourfold.
    ///
    /// An isotropic stand-in for the traced part keeps this measuring the lobe and not the Monte
    /// Carlo noise of a second ray-tracing run.
    #[test]
    fn an_unresolved_lobe_keeps_its_energy() {
        let isotropic = RayTracingOptics {
            single_scattering_albedo: 1.0,
            asymmetry: 0.0,
            rays: 0,
        };
        let mut asymmetries = Vec::new();
        for steps in [720usize, 360, 180] {
            let mu: Vec<f64> = (0..=steps)
                .map(|i| {
                    libm::cos(core::f64::consts::PI * (steps - i) as f64 / steps as f64)
                        .clamp(-1.0, 1.0)
                })
                .collect();
            let mut value = vec![0.5; mu.len()];
            let optics = add_diffraction(&cirrus(), &isotropic, &mu, &mut value).unwrap();
            asymmetries.push(optics.asymmetry);
        }
        let spread = asymmetries
            .iter()
            .fold(0.0_f64, |a, g| a.max((g - asymmetries[0]).abs()));
        assert!(spread < 1e-3, "asymmetry across grids: {asymmetries:?}");
        // Half the energy is in the lobe and half is isotropic, so the combined `g` is half the
        // lobe's — which the Airy tail keeps just under 1.
        assert!(
            asymmetries[0] > 0.49 && asymmetries[0] < 0.5,
            "{asymmetries:?}"
        );
    }

    #[test]
    fn a_lobe_cannot_be_built_on_a_grid_that_is_not_one() {
        let crystal = cirrus();
        let mut out = vec![0.0; 3];
        assert_eq!(
            diffraction_on_grid(&crystal, &[-1.0, 0.0, 1.0], &mut out[..2]),
            None
        );
        assert_eq!(diffraction_on_grid(&crystal, &[1.0], &mut out[..1]), None);
        let flat = HexagonalCrystal {
            aspect_ratio: 0.0,
            ..crystal
        };
        assert_eq!(
            diffraction_on_grid(&flat, &[-1.0, 0.0, 1.0], &mut out),
            None
        );

        let empty = RayTracingOptics {
            single_scattering_albedo: 1.0,
            asymmetry: 0.0,
            rays: 0,
        };
        let mu = [-1.0, 0.0, 1.0];
        let mut zero = vec![0.0; 3];
        assert_eq!(add_diffraction(&crystal, &empty, &mu, &mut zero), None);

        // A grid with a repeated node passes every length and parameter check and still is not a
        // grid: the cell of zero width divides by its own zero and the lobe comes back `NaN`. That
        // is caught on the lobe's integral rather than on the grid, and it has to be caught there
        // for the right reason — `NaN <= 0.0` is false, so a guard that tested only the sign would
        // let it through and hand the caller a phase function of `NaN` reported as success.
        let repeated = [-1.0, 0.0, 0.0, 1.0];
        let mut flat_ray = vec![1.0; 4];
        assert_eq!(
            add_diffraction(&crystal, &empty, &repeated, &mut flat_ray),
            None
        );
    }

    /// Roughening the facets erases the halos — the published behaviour, from the geometry.
    ///
    /// This is what stands between a smooth prism and the habits the operational bulk models are
    /// built from. The measurement is the ratio `p(22°)/p(21°)`, the sharp inner edge that makes a
    /// halo a halo: **31.3** on a perfect prism, **1.86** at Yang's *moderately roughened* `σ = 0.03`,
    /// **1.03** at `σ = 0.15` and **0.90** at *severely roughened* `σ = 0.50`, where the phase
    /// function simply decreases through the halo angles. The 46° feature goes the same way, 2.53 to
    /// 0.98.
    ///
    /// Note what does *not* move much: the asymmetry parameter, 0.628 to 0.600. Roughening rearranges
    /// where the light goes without changing how forward it is on average, which is exactly why a
    /// one-parameter fit to `g` cannot tell the two habits apart and this module exists.
    #[test]
    fn roughening_erases_the_halos() {
        let mu = angle_grid();
        let mut contrast = Vec::new();
        for roughness in [0.0, 0.03, 0.15, 0.50] {
            let crystal = HexagonalCrystal {
                roughness,
                ..cirrus()
            };
            let mut value = vec![0.0; mu.len()];
            let optics = trace_on_grid(&crystal, &mu, &mut value, 400_000, 7).unwrap();
            let mut cumulative = vec![0.0; mu.len()];
            TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).unwrap();
            contrast.push((
                at_degrees(&mu, &value, 22.0) / at_degrees(&mu, &value, 21.0),
                at_degrees(&mu, &value, 46.0) / at_degrees(&mu, &value, 45.0),
                optics.asymmetry,
            ));
        }
        assert!(
            contrast[0].0 > 25.0,
            "the smooth prism has a 22° halo: {contrast:?}"
        );
        assert!(contrast[0].1 > 2.0, "and a 46° one: {contrast:?}");
        for (previous, next) in contrast.iter().zip(contrast.iter().skip(1)) {
            assert!(
                next.0 < previous.0,
                "the halo only ever weakens: {contrast:?}"
            );
        }
        assert!(
            contrast[1].0 < 2.0,
            "moderate roughening flattens it: {contrast:?}"
        );
        assert!(
            contrast[3].0 < 1.0,
            "severe roughening inverts the slope: {contrast:?}"
        );
        assert!(
            contrast[3].1 < 1.0,
            "and the 46° feature with it: {contrast:?}"
        );
        for c in &contrast {
            assert!((0.55..0.70).contains(&c.2), "g barely moves: {contrast:?}");
        }
    }

    /// (G4) A transparent crystal returns everything it intercepts **at every roughness**.
    ///
    /// The regression this pins: a micro-facet tilted far enough can reflect a ray straight through
    /// the face it is sitting on, or refract one back the way it came. Those rays leave the geometry
    /// and the tracer stops following them, which does not look like a bug — it looks like a
    /// single-scattering albedo of 0.77 on ice with no absorption in it. Rejecting the inconsistent
    /// draws restores it to 1 exactly.
    #[test]
    fn a_transparent_crystal_conserves_energy_however_rough_it_is() {
        let mu = angle_grid();
        for roughness in [0.0, 0.03, 0.15, 0.50, 1.0] {
            let crystal = HexagonalCrystal {
                roughness,
                ..cirrus()
            };
            let mut value = vec![0.0; mu.len()];
            let optics = trace_on_grid(&crystal, &mu, &mut value, 100_000, 13).unwrap();
            assert!(
                optics.single_scattering_albedo > 1.0 - 1e-4,
                "σ = {roughness} lost {} of the light",
                1.0 - optics.single_scattering_albedo
            );
        }
    }
}
