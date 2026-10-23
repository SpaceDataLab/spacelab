//! The ice ray tracer against a published parameterisation, number for number.
//!
//! Everything else about this crystal is anchored on physics it derives itself — Snell's minimum
//! deviation puts the halo at 21.8°, the zero of `J₁` puts the diffraction ring at 0.72°, energy is
//! conserved. None of that says whether the *amount* of forward scattering is right, and the
//! asymmetry parameter is the one number a radiative-transfer solver actually consumes.
//!
//! # The oracle
//!
//! Fu, Q. (2007), "A New Parameterization of an Asymmetry Factor of Cirrus Clouds for Climate
//! Models", *J. Atmos. Sci.* **64**(11), 4140–4150, Tables 1 and 2, band `0.25–0.70 µm`. The
//! coefficients below are a numerical fit to geometric ray-tracing calculations (Takano & Liou 1989;
//! Yang & Liou 1995) for randomly oriented hexagonal ice crystals over 28 in-situ cirrus size
//! distributions, with a stated average absolute relative error of **0.4%** against those reference
//! calculations. It is an independent implementation by an independent group; nothing in it shares a
//! line with this workspace.
//!
//! Fu parameterises two things in terms of the aspect ratio `AR = D/L` (width over length):
//!
//! * `g′`, the asymmetry factor of the **geometric** part alone — reflection and refraction, with
//!   diffraction excluded. That is exactly [`RayTracingOptics::asymmetry`].
//! * `f_d`, the **delta-transmission** fraction: light passing undeviated through the parallel faces
//!   of the crystal, as a fraction of the *total* scattered energy.
//!
//! and combines them with his Eq. (2.2), `g = 1/(2ω̃) + [1 − 1/(2ω̃)]·g′`, which is the same
//! two-component split [`add_diffraction`] implements — derived here from `Q_ext → 2` rather than
//! quoted, and shown below to agree with his form identically.
//!
//! Transcribing coefficients and comparing numbers is what a published *result* is for; the paper's
//! prose and figures are its authors'. The values are cited, not reproduced in bulk.
//!
//! # The aspect-ratio convention, and how it is confirmed rather than assumed
//!
//! Fu's `AR = D/L` with `D` the width across the hexagon's vertices and `L` the length;
//! [`HexagonalCrystal::aspect_ratio`] is the half-length over the circumradius, `(L/2)/a`, and
//! `D = 2a`, so `AR = 1/aspect_ratio`. That mapping is not taken on trust: Fu states the asymmetry
//! factor has its **minimum at `D/L ≈ 1`**, and the sweep below finds this tracer's minimum at
//! `AR = 1.0` — which it would not, under a different convention.

// Test code, and the transcribed coefficients are quoted at the precision the paper prints them.
#![allow(
    clippy::unreadable_literal,
    // The sweep's aspect ratios are the literals declared above, so identifying the minimum by
    // equality against one of them is exact, not a tolerance question.
    clippy::float_cmp,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss
)]

use spacelab_earthvision::ice::{HexagonalCrystal, add_diffraction, trace_on_grid};
use spacelab_earthvision::phase::TabulatedPhase;

/// Fu (2007) Table 1, `g′_s`, band 0.25–0.70 µm, for `0.1 ≤ AR ≤ 1`: `cs0 + cs1·AR + cs2·AR²`.
const SMOOTH_QUADRATIC: [f64; 3] = [7.938_904e-1, -3.987_320e-1, 1.349_959e-1];
/// Fu (2007) Table 2, `g′_s`, same band, for `1 < AR ≤ 20`: `ps0 + ps1·ln AR + ps2·ln² AR`.
const SMOOTH_LOGARITHMIC: [f64; 3] = [5.292_852e-1, 1.140_557e-1, 3.165_543e-3];
/// Fu (2007) Table 1, `g′_r` — his *rough* crystal, `0.1 ≤ AR ≤ 1`.
const ROUGH_QUADRATIC: [f64; 3] = [7.386_498e-1, -4.157_560e-1, 1.425_497e-1];
/// Fu (2007) Table 2, `g′_r`, `1 < AR ≤ 20`.
const ROUGH_LOGARITHMIC: [f64; 3] = [4.653_916e-1, 7.070_839e-2, 1.659_297e-2];
/// Fu (2007) Eq. (3.5) with `dc0 = 1`, `dc1 = 0` in the visible: `f_d` for `0.1 ≤ AR ≤ 1`.
const DELTA_QUADRATIC: [f64; 3] = [0.1887, -0.1157, 0.048_44];
/// Fu (2007) Eq. (3.6) with `dp0 = 1`, `dp1 = 0`: `f_d` for `1 < AR ≤ 20`.
const DELTA_LOGARITHMIC: [f64; 3] = [0.106, 0.0974, -0.000_165];

/// The aspect ratios swept, `D/L`, spanning Fu's whole validity range bar its ends.
const ASPECT_RATIOS: [f64; 9] = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0];
const RAYS: u64 = 200_000;
const SEED: u64 = 7;
/// Ice at 550 nm.
const REFRACTIVE_INDEX: f64 = 1.3105;
/// Yang et al. (2013) call this *severely roughened*.
const SEVERE: f64 = 0.5;

fn evaluate(quadratic: [f64; 3], logarithmic: [f64; 3], aspect: f64) -> f64 {
    if aspect <= 1.0 {
        quadratic[0] + quadratic[1] * aspect + quadratic[2] * aspect * aspect
    } else {
        let l = aspect.ln();
        logarithmic[0] + logarithmic[1] * l + logarithmic[2] * l * l
    }
}

fn fu_smooth(aspect: f64) -> f64 {
    evaluate(SMOOTH_QUADRATIC, SMOOTH_LOGARITHMIC, aspect)
}

fn fu_rough(aspect: f64) -> f64 {
    evaluate(ROUGH_QUADRATIC, ROUGH_LOGARITHMIC, aspect)
}

fn fu_delta(aspect: f64) -> f64 {
    evaluate(DELTA_QUADRATIC, DELTA_LOGARITHMIC, aspect)
}

/// A quarter-degree grid in scattering angle, ascending in cosine.
fn angle_grid() -> Vec<f64> {
    (0..721)
        .map(|i| {
            (std::f64::consts::PI * f64::from(720 - i) / 720.0)
                .cos()
                .clamp(-1.0, 1.0)
        })
        .collect()
}

struct Traced {
    /// Ray-tracing asymmetry parameter — Fu's `g′`.
    geometric_asymmetry: f64,
    /// Delta-transmission fraction, as a share of *total* scattered energy — Fu's `f_d`.
    delta_fraction: f64,
    /// Asymmetry parameter of both components together — Fu's `g`.
    total_asymmetry: f64,
    single_scattering_albedo: f64,
}

fn trace(aspect: f64, roughness: f64) -> Traced {
    let crystal = HexagonalCrystal {
        aspect_ratio: 1.0 / aspect,
        refractive_index: REFRACTIVE_INDEX,
        absorption: 0.0,
        size_parameter: 200.0,
        roughness,
    };
    let mu = angle_grid();
    let mut value = vec![0.0; mu.len()];
    let rays = trace_on_grid(&crystal, &mu, &mut value, RAYS, SEED).unwrap();
    let mut cumulative = vec![0.0; mu.len()];
    TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).unwrap();

    // The undeviated beam, within one degree of forward. The last four cells rather than the last
    // one: reconstructing node values from interval means spreads a sub-cell spike across the two
    // cells that meet at the peak, and by 1° the tally has stopped growing — 0.2781 against 0.2788
    // out to 2° at `AR = 0.5`, so this is the delta transmission and not a widening window.
    let last = mu.len() - 1;
    let mut peak = 0.0;
    for i in last - 4..last {
        peak += f64::midpoint(value[i], value[i + 1]) * (mu[i + 1] - mu[i]);
    }

    let optics = add_diffraction(&crystal, &rays, &mu, &mut value).unwrap();
    Traced {
        geometric_asymmetry: rays.asymmetry,
        // Fu normalises by the total scattered energy, of which the traced part is `ω_ray/(1+ω_ray)`.
        delta_fraction: peak * rays.single_scattering_albedo
            / (1.0 + rays.single_scattering_albedo),
        total_asymmetry: optics.asymmetry,
        single_scattering_albedo: optics.single_scattering_albedo,
    }
}

/// The geometric asymmetry parameter and the delta-transmission fraction, against Fu (2007).
///
/// Measured over two decades of aspect ratio, worst case in each column:
///
/// ```text
///     AR      g′ ours   g′ Fu     Δ        f_d ours  f_d Fu    Δ
///     0.10    0.7500    0.7554    −0.7%    0.1727    0.1776    −2.8%
///     0.50    0.6287    0.6283    +0.1%    0.1391    0.1430    −2.7%
///     1.00    0.5414    0.5302    +2.1%    0.1202    0.1214    −1.0%
///     2.00    0.5841    0.6099    −4.2%    0.1643    0.1734    −5.3%
///     10.0    0.8287    0.8087    +2.5%    0.3454    0.3294    +4.9%
/// ```
///
/// Two things make that agreement meaningful rather than lucky. Fu's fit carries its own 0.4% error
/// against the ray tracing it was fitted to, and it is fitted to *size distributions* of crystals with
/// a mean effective aspect ratio, where this is one monodisperse crystal — so a few percent is the
/// floor, not the ceiling, of what the comparison can resolve. And the Monte Carlo's own repeatability
/// was measured across seeds at `1–2e-3` in `g′`, an order below the residuals reported.
#[test]
fn the_ray_traced_asymmetry_and_delta_transmission_match_fu_2007() {
    let mut worst_asymmetry: f64 = 0.0;
    let mut worst_delta: f64 = 0.0;
    for aspect in ASPECT_RATIOS {
        let ours = trace(aspect, 0.0);
        let asymmetry = ours.geometric_asymmetry / fu_smooth(aspect) - 1.0;
        let delta = ours.delta_fraction / fu_delta(aspect) - 1.0;
        assert!(
            asymmetry.abs() < 0.055,
            "AR = {aspect}: g′ = {} against Fu's {}, {:.2}%",
            ours.geometric_asymmetry,
            fu_smooth(aspect),
            100.0 * asymmetry
        );
        assert!(
            delta.abs() < 0.08,
            "AR = {aspect}: f_d = {} against Fu's {}, {:.2}%",
            ours.delta_fraction,
            fu_delta(aspect),
            100.0 * delta
        );
        worst_asymmetry = worst_asymmetry.max(asymmetry.abs());
        worst_delta = worst_delta.max(delta.abs());
    }
    // Guard against the tolerances quietly becoming meaningless: the measured worst cases are 4.2%
    // and 5.3%, so a suite that suddenly agrees to 1% everywhere has changed the oracle, not the code.
    assert!(
        worst_asymmetry > 0.01,
        "suspiciously good: {worst_asymmetry}"
    );
    assert!(worst_delta > 0.01, "suspiciously good: {worst_delta}");
}

/// The *combined* asymmetry parameter lands in the published range, with the published shape.
///
/// Fu states the asymmetry factor of hexagonal ice crystals in the visible "rang[es] from 0.77 to
/// above 0.9 with the minimum value when D/L is about one". This tracer, with diffraction included:
/// **0.771** at `AR = 1` rising to **0.914** at `AR = 10`, minimum at `AR = 1`. The location of that
/// minimum is also what confirms the `AR = 1/aspect_ratio` mapping — get the convention wrong and the
/// minimum lands somewhere else.
#[test]
fn the_total_asymmetry_spans_the_published_range_with_its_minimum_at_unit_aspect_ratio() {
    let measured: Vec<(f64, f64)> = ASPECT_RATIOS
        .iter()
        .map(|aspect| (*aspect, trace(*aspect, 0.0).total_asymmetry))
        .collect();
    for (aspect, g) in &measured {
        assert!(
            (0.765..0.925).contains(g),
            "AR = {aspect}: g = {g} outside Fu's published 0.77…>0.9"
        );
        let relative = g / (0.5 + 0.5 * fu_smooth(*aspect)) - 1.0;
        assert!(
            relative.abs() < 0.025,
            "AR = {aspect}: g = {g} against Fu's {}, {:.2}%",
            0.5 + 0.5 * fu_smooth(*aspect),
            100.0 * relative
        );
    }
    let minimum = measured
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    assert_eq!(minimum.0, 1.0, "the minimum is at D/L ≈ 1: {measured:?}");
    let maximum = measured
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    assert!(
        maximum.1 > 0.9,
        "and the thinnest plate is above 0.9: {measured:?}"
    );
}

/// Roughening destroys the delta transmission, which is what Fu says it does — and then the two
/// models part company by a measured amount, for a reason worth writing down.
///
/// Fu: "The ice particles with rough surfaces should not have the delta transmission associated with
/// light beams passing through the parallel planes of ice crystals." At `σ = 0.5` this tracer's
/// delta-transmission fraction is **below 0.001** at every aspect ratio, from 0.12–0.35 smooth. The
/// facets are no longer parallel, so nothing passes through undeviated. That is the physics, arrived
/// at by perturbing normals rather than by asserting it.
///
/// Where the two part company is what happens to the *rest* of the light. Fu's rough asymmetry factor
/// is a proxy — he removes the delta transmission and assumes the remainder keeps its asymmetry,
/// `g_r = (g_s − f_d)/(1 − f_d)`. Run that algebra on *this* tracer's own `g_s` and `f_d` and it
/// reproduces his `g_r` to **1.7%** across the sweep, which is a joint check on both inputs. Run the
/// physical roughening instead and the answer is up to **7.3% higher** near `AR = 1`, because tilting
/// facets does not merely delete the forward spike — it scatters some of that light into the small
/// angles around it, which the proxy has no way to represent. Fu says as much: "we simply represent
/// the effect of ice particle surface roughness on g by removing the contribution of the delta
/// transmission". The divergence is bounded here rather than hidden.
#[test]
fn roughening_removes_the_delta_transmission_and_then_diverges_from_fus_proxy() {
    let mut worst_proxy: f64 = 0.0;
    let mut worst_physical: f64 = 0.0;
    for aspect in ASPECT_RATIOS {
        let smooth = trace(aspect, 0.0);
        let rough = trace(aspect, SEVERE);
        assert!(
            rough.delta_fraction < 0.005,
            "AR = {aspect}: a severely roughened crystal still transmits {} undeviated",
            rough.delta_fraction
        );

        let published = 0.5 + 0.5 * fu_rough(aspect);
        let proxy =
            (smooth.total_asymmetry - smooth.delta_fraction) / (1.0 - smooth.delta_fraction);
        let by_proxy = proxy / published - 1.0;
        let by_physics = rough.total_asymmetry / published - 1.0;
        assert!(
            by_proxy.abs() < 0.025,
            "AR = {aspect}: Fu's own algebra on our numbers gives {proxy} against his {published}"
        );
        assert!(
            by_physics.abs() < 0.08,
            "AR = {aspect}: physical roughening gives {} against Fu's {published}",
            rough.total_asymmetry
        );
        worst_proxy = worst_proxy.max(by_proxy.abs());
        worst_physical = worst_physical.max(by_physics.abs());
    }
    // The physical model is *supposed* to disagree with the proxy more than it disagrees with the
    // reference calculation. If that ordering ever inverts, one of the two has stopped being itself.
    assert!(
        worst_physical > worst_proxy,
        "physics {worst_physical} vs proxy {worst_proxy}"
    );
}

/// The two-component combination is Fu's Eq. (2.2), and the visible albedo is his 0.50.
///
/// `add_diffraction` weights the components by the scattering efficiencies geometric optics gives
/// them, `1` and `ω_ray` out of `Q_ext = 2`. Fu writes the same split from the other end:
/// `g = 1/(2ω̃) + [1 − 1/(2ω̃)]·g′`, where `1/(2ω̃)` is the diffracted share of the scattered energy.
/// The two are the same statement — `1/(1 + ω_ray) = 1/(2ω̃)` — and this checks it numerically.
///
/// The residual is not zero and should not be: Fu takes the diffraction lobe's own asymmetry as
/// exactly 1, where this computes it from the Airy pattern and gets `0.9983` at `x = 200`. That
/// difference is `(1 − g_diff)/(2ω̃) ≈ 8.5e-4`, and it is the amount by which the model here is more
/// precise than the identity it is being checked against.
///
/// Fu also notes that `1/(2ω̃)` is 0.50 for wavelengths below about 1.4 µm — ice does not absorb
/// there — which is `ω̃ = 1` exactly, and the transparent crystal traced here reproduces it.
#[test]
fn the_combination_is_fu_equation_2_2() {
    let ours = trace(0.5, 0.0);
    assert!(
        (ours.single_scattering_albedo - 1.0).abs() < 1e-4,
        "ω̃ = {} in the visible, so Fu's 1/(2ω̃) is 0.50",
        ours.single_scattering_albedo
    );

    let diffracted_share = 1.0 / (2.0 * ours.single_scattering_albedo);
    let fu = diffracted_share + (1.0 - diffracted_share) * ours.geometric_asymmetry;
    let residual = ours.total_asymmetry - fu;
    assert!(
        residual.abs() < 2e-3,
        "Eq. (2.2) gives {fu}, we give {}",
        ours.total_asymmetry
    );
    // Signed, and on the side the approximation predicts: `g_diff < 1` can only pull the answer down.
    assert!(residual < 0.0, "residual {residual} has the wrong sign");
}
