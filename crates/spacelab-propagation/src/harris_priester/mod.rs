//! Modified Harris-Priester empirical thermospheric density model.
//!
//! The static Harris-Priester model of Montenbruck & Gill, "Satellite Orbits" (Springer 2005), §3.5:
//! a min/max density-vs-altitude table for mean solar activity (`tables::ALT_RHO`, 100–1000 km),
//! blended by the diurnal-bulge cosine law `ρ = ρ_min + (ρ_max − ρ_min)·cosⁿ(ψ/2)`. `ψ` is the angle
//! between the satellite and the apex of the diurnal density bulge, which lags the sub-solar point by
//! `LAG` = 30° east; the cosine exponent `n` runs from ~2 for low-inclination orbits to ~6 for polar
//! ones. Densities within a table bracket are interpolated exponentially. It needs no space-weather
//! data — only the Sun's direction — which is what makes it the cheap, robust default for coarse drag.
//!
//! The public surface is the pure [`harris_priester`] function, whose explicit-vector interface
//! (Sun and satellite positions in a common Earth-centred frame) mirrors Orekit's
//! `HarrisPriester.getDensity(sunInEarth, posInEarth)`, plus the [`HarrisPriester`] model
//! configuration (ellipsoid + cosine exponent) and the [`HarrisPriesterAtmosphere`] wrapper that
//! adapts it to a [`crate::forces::Atmosphere`] for the drag path.
//!
//! The 30° bulge lag is taken at full precision (`π/6`, not a 7-digit `toRadians(30)` literal),
//! matching this crate's constant policy. The density table and the interpolation/cosine algorithm
//! are validated against Orekit 13.1.7's `HarrisPriester.getDensity` — an independent Java lineage of
//! the same Montenbruck & Gill model — to a few ULP (see the module tests).

#![allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::must_use_candidate,
    clippy::similar_names,
    // The half-angle cosine is kept as `(1 + cos ψ)/2` verbatim from Orekit for bit-for-bit parity;
    // `f64::midpoint` rounds differently. The above-ceiling density is exactly `0.0` by definition.
    clippy::manual_midpoint,
    clippy::float_cmp,
    clippy::doc_markdown
)]

mod atmosphere;
mod tables;

pub use atmosphere::HarrisPriesterAtmosphere;
use core::f64::consts::FRAC_PI_6;
use tables::ALT_RHO;

/// Lag angle of the diurnal-bulge apex east of the sub-solar point: 30° = `π/6` exactly
/// (Montenbruck & Gill §3.5), taken at full precision rather than a truncated `toRadians(30)` value.
const LAG: f64 = FRAC_PI_6;

/// Minimum `cos(ψ/2)` below which the diurnal term is taken as identically zero — the satellite is on
/// the night side, so only the minimum (background) density survives. Matches Orekit's `MIN_COS`.
const MIN_COS: f64 = 1.0e-12;

/// Configuration of the Modified Harris-Priester model: the reference ellipsoid used to reduce a
/// position to an altitude, and the diurnal-bulge cosine exponent. The min/max density table is the
/// fixed mean-activity table of Montenbruck & Gill (`tables::ALT_RHO`); the caller states the
/// ellipsoid and exponent explicitly, never a silent default (the crate's constant policy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarrisPriester {
    /// Ellipsoid equatorial radius, m (e.g. WGS84 `6_378_137.0`).
    pub equatorial_radius: f64,
    /// Ellipsoid flattening (e.g. WGS84 `1.0 / 298.257_223_563`).
    pub flattening: f64,
    /// Diurnal-bulge cosine exponent `n`: ~2 for low-inclination orbits, ~6 for polar ones.
    pub n: f64,
}

/// Altitude above the reference ellipsoid, m — Montenbruck & Gill's closed-form approximation valid
/// for the atmosphere layer (Orekit `HarrisPriester.getHeight`). `e² = f(2 − f)`; on the equator this
/// reduces to `|r| − a`, at the pole to `|r| − b`.
fn height(pos: [f64; 3], a: f64, f: f64) -> f64 {
    let e2 = f * (2.0 - f);
    let r = libm::sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
    let sl = pos[2] / r; // sin(geocentric latitude)
    let cl2 = 1.0 - sl * sl;
    let coef = libm::sqrt((1.0 - e2) / (1.0 - e2 * cl2));
    r - a * coef
}

/// Unit vector, or the zero vector unchanged if the input is null.
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Total mass density `ρ` (kg/m³) at the satellite, from the Sun and satellite positions in a common
/// Earth-centred frame (both metres). Mirrors Orekit's `HarrisPriester.getDensity(sunInEarth,
/// posInEarth)`: reduce the position to an altitude, form the diurnal-bulge apex (Sun direction turned
/// `LAG` east about `+z`), blend the exponentially-interpolated min/max densities by `cosⁿ(ψ/2)`.
///
/// Above the table ceiling (1000 km) the density is zero. Below the floor (100 km) the model is out of
/// its validity range; rather than fault the propagator mid-step, the lowest bracket is extrapolated
/// (density rising) — a re-entering object is braking hard there anyway. On the night side, where
/// `cos(ψ/2)` collapses below `MIN_COS`, only the minimum density survives.
pub fn harris_priester(
    sun_in_earth: [f64; 3],
    pos_in_earth: [f64; 3],
    model: &HarrisPriester,
) -> f64 {
    let pos_alt = height(pos_in_earth, model.equatorial_radius, model.flattening);
    let top = ALT_RHO.len() - 1;
    if pos_alt > ALT_RHO[top][0] {
        return 0.0;
    }

    // Diurnal-bulge apex: the Sun direction rotated LAG east about +z, then re-normalised.
    let sun_dir = normalize(sun_in_earth);
    let (sin_lag, cos_lag) = (libm::sin(LAG), libm::cos(LAG));
    let bulge = normalize([
        sun_dir[0] * cos_lag - sun_dir[1] * sin_lag,
        sun_dir[0] * sin_lag + sun_dir[1] * cos_lag,
        sun_dir[2],
    ]);
    let pos_dir = normalize(pos_in_earth);
    let cos_psi = bulge[0] * pos_dir[0] + bulge[1] * pos_dir[1] + bulge[2] * pos_dir[2];

    // cosⁿ(ψ/2) via cos²(ψ/2) = (1 + cos ψ)/2, guarded to zero on the night side.
    let c2psi2 = (1.0 + cos_psi) / 2.0;
    let cpsi2 = libm::sqrt(c2psi2);
    let cos_pow = if cpsi2 > MIN_COS {
        c2psi2 * libm::pow(cpsi2, model.n - 2.0)
    } else {
        0.0
    };

    // Locate the altitude bracket [h_ia, h_ia+1] (linear scan; the table has 50 rows).
    let mut ia = 0;
    while ia < ALT_RHO.len() - 2 && pos_alt > ALT_RHO[ia + 1][0] {
        ia += 1;
    }
    // Fractional height: 0 at the lower node, 1 at the upper.
    let dh = (ALT_RHO[ia][0] - pos_alt) / (ALT_RHO[ia][0] - ALT_RHO[ia + 1][0]);

    let rho_min = ALT_RHO[ia][1] * libm::pow(ALT_RHO[ia + 1][1] / ALT_RHO[ia][1], dh);
    if cos_pow == 0.0 {
        return rho_min;
    }
    let rho_max = ALT_RHO[ia][2] * libm::pow(ALT_RHO[ia + 1][2] / ALT_RHO[ia][2], dh);
    rho_min + (rho_max - rho_min) * cos_pow
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::Atmosphere;
    use spacelab_autodiff::Dual;
    use spacelab_testkit::nan_max;

    /// WGS84, the ellipsoid the Orekit oracle was built with.
    const WGS84_A: f64 = 6_378_137.0;
    const WGS84_F: f64 = 1.0 / 298.257_223_563;

    /// Sun position (m) in the Earth frame shared by every golden case: right ascension 20°,
    /// declination −15°, at one astronomical unit — the value the Orekit oracle emitted.
    const SUN: [f64; 3] = [
        1.35786003621169880e+11,
        4.94220635480685960e+10,
        -3.87187780439443740e+10,
    ];

    /// Orekit 13.1.7 `HarrisPriester.getDensity(sunInEarth, posInEarth)` densities (independent Java
    /// lineage of the same Montenbruck & Gill model, same full-precision 30° lag). Each row is
    /// `(satellite position in the Earth frame [m], cosine exponent n, density [kg/m³])`. The sweep
    /// spans 100–1000 km across every table-bracket regime, exponents 2/4/6, and bulge angles from the
    /// apex to its exact antipode (row 11 — the night-side `cosⁿ(ψ/2) → 0` branch, returning ρ_min).
    /// Generated by `tools/orekit-oracle/OrekitPropagationBench.writeHarrisPriesterValidation`.
    #[rustfmt::skip]
    const GOLDEN: [([f64; 3], f64, f64); 16] = [
        ([ 4.17049407354879430e+06,  4.97020129504463600e+06,  0.00000000000000000e+00], 4.0, 1.11289083022565400e-07),
        ([ 4.12929067527690060e+06,  4.92109699712701700e+06,  1.13273085272172700e+06], 4.0, 2.87226390866814900e-09),
        ([ 4.08426759779761820e+06,  4.86744058279143500e+06, -1.70254713689356090e+06], 2.0, 3.03824276957614100e-10),
        ([ 6.22841142845804900e+06,  2.26695636672216800e+06,  0.00000000000000000e+00], 4.0, 8.74366972247165500e-11),
        ([-4.43036923358306850e+06,  3.71752118987872300e+06,  3.33906850000000000e+06], 6.0, 1.65777277216442630e-11),
        ([ 4.32476309987356400e+06,  5.15405196139319150e+06,  0.00000000000000000e+00], 4.0, 1.52673219138190130e-11),
        ([ 4.20844462840058750e+06,  5.01542900513020400e+06, -1.75431094591406500e+06], 4.0, 7.34456890986249500e-12),
        ([-3.10352126271315600e+06, -3.69863261453092100e+06,  4.82822197557076900e+06], 6.0, 7.61476924175720000e-13),
        ([-2.35246140255358500e+06,  6.46333458365452600e+06,  0.00000000000000000e+00], 2.0, 1.61534099592686960e-12),
        ([ 3.88452230964664840e+06,  4.62939341803375100e+06, -3.48906850000000000e+06], 4.0, 5.84384927699522600e-13),
        ([-4.39471017430504100e+06, -5.23741163863835800e+06,  1.83195665944482130e+06], 4.0, 2.00801100137866800e-14),
        ([ 4.33575821969815300e+06,  5.16715543494514900e+06,  2.45506744555128550e+06], 4.0, 6.64583444987492000e-14),
        ([ 3.61406849999999900e+06,  6.25975026403424100e+06,  0.00000000000000000e+00], 6.0, 4.96393082840847550e-14),
        ([ 4.51888720490801000e+06,  5.38540006097712800e+06, -1.88372046846532540e+06], 2.0, 3.59154296088721300e-14),
        ([ 3.60840306636781250e+06, -6.24993744513638000e+06,  1.27251763574360600e+06], 4.0, 4.02380117388072250e-15),
        ([ 4.73614717007294950e+06,  5.64432040498933800e+06,  0.00000000000000000e+00], 4.0, 1.87298589064156980e-14),
    ];

    /// The port reproduces Orekit's density across the whole sweep to a few ULP — the two share the
    /// Montenbruck & Gill table and the full-precision 30° lag, so only `pow`/`sinCos` rounding differs
    /// (libm vs Hipparchus). A wrong table or a mis-transcribed cosine law would miss by orders of
    /// magnitude, not `1e-12`.
    #[test]
    fn matches_orekit_reference() {
        let mut max_rel = 0.0_f64;
        for &(pos, n, rho_ref) in &GOLDEN {
            let model = HarrisPriester {
                equatorial_radius: WGS84_A,
                flattening: WGS84_F,
                n,
            };
            let rho = harris_priester(SUN, pos, &model);
            max_rel = nan_max(max_rel, ((rho - rho_ref) / rho_ref).abs());
        }
        assert!(
            max_rel < 1.0e-11,
            "max density rel err vs Orekit {max_rel:e} exceeds 1e-11"
        );
    }

    /// Above the 1000 km table ceiling the density is exactly zero (Orekit returns 0 there too).
    #[test]
    fn above_ceiling_is_zero() {
        let model = HarrisPriester {
            equatorial_radius: WGS84_A,
            flattening: WGS84_F,
            n: 4.0,
        };
        let r = WGS84_A + 1_050_000.0;
        assert_eq!(harris_priester(SUN, [r, 0.0, 0.0], &model), 0.0);
    }

    /// The ceiling is closed from below: `> ceiling` returns zero, so *at* 1000 km the model still
    /// interpolates, landing on the last table row. `above_ceiling_is_zero` probes 1050 km and cannot
    /// tell that test from `>=`, which would open the interval and drop the density to zero one row
    /// early. On the equator the ellipsoid reduction is exactly `|r| − a` and both radii are integers
    /// below 2⁵³, so the altitude here is exactly the tabulated ceiling, not a value near it.
    #[test]
    fn at_the_ceiling_the_density_is_the_last_table_row() {
        let model = HarrisPriester {
            equatorial_radius: WGS84_A,
            flattening: WGS84_F,
            n: 4.0,
        };
        let at_ceiling = [WGS84_A + ALT_RHO[ALT_RHO.len() - 1][0], 0.0, 0.0];
        let one_metre_below = [at_ceiling[0] - 1.0, 0.0, 0.0];
        assert_eq!(
            height(at_ceiling, WGS84_A, WGS84_F),
            ALT_RHO[ALT_RHO.len() - 1][0],
            "the fixture is not exactly at the ceiling"
        );
        let rho = harris_priester(SUN, at_ceiling, &model);
        let below = harris_priester(SUN, one_metre_below, &model);
        let jump = ((rho - below) / below).abs();
        println!(
            "HP at the 1000 km ceiling: ρ = {rho:e}, one metre below {below:e} (jump {jump:e})"
        );
        assert!(rho > 0.0, "the ceiling was excluded: ρ = {rho:e}");
        // One metre against a ~100 km scale height: continuous to about 1e-5, nowhere near a step to 0.
        assert!(
            jump < 1.0e-4,
            "the density steps at the ceiling: {below:e} → {rho:e}"
        );
    }

    /// At the exact diurnal-bulge antipode the cosine power vanishes, so only ρ_min survives — the
    /// density is independent of the exponent `n` there.
    #[test]
    fn night_side_is_exponent_independent() {
        let antipode = GOLDEN[10].0; // row 11: exact antipode of the apex
        let base = HarrisPriester {
            equatorial_radius: WGS84_A,
            flattening: WGS84_F,
            n: 4.0,
        };
        let steep = HarrisPriester { n: 6.0, ..base };
        let d4 = harris_priester(SUN, antipode, &base);
        let d6 = harris_priester(SUN, antipode, &steep);
        assert!(
            ((d4 - d6) / d4).abs() < 1.0e-15,
            "night-side density depends on n: {d4:e} vs {d6:e}"
        );
    }

    /// The [`HarrisPriesterAtmosphere`] wrapper, evaluated in f64 at exactly a reference point, returns
    /// the bare model density (the local-exponential lift factor is `exp(0) = 1` there).
    #[test]
    fn wrapper_reduces_to_model_at_point() {
        let (pos, n, rho_ref) = GOLDEN[6]; // ~400 km, off-equator, apex-side
        let model = HarrisPriester {
            equatorial_radius: WGS84_A,
            flattening: WGS84_F,
            n,
        };
        let atmo = HarrisPriesterAtmosphere {
            model,
            geometry: |_t: f64, _r: [f64; 3]| pos,
            sun: |_t: f64| SUN,
        };
        // Place the inertial position at |pos| so the lift exponent is exactly zero.
        let r_mag = libm::sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
        let rho = atmo.density(0.0, &[r_mag, 0.0, 0.0]);
        assert!(((rho - rho_ref) / rho_ref).abs() < 1.0e-11);
    }

    /// The wrapper's `Scalar`-generic density produces a finite, negative radial gradient in `Dual`
    /// (density falls with altitude), so it drops into the STM / dd-precision paths unchanged.
    #[test]
    fn wrapper_is_generic_over_scalar() {
        let model = HarrisPriester {
            equatorial_radius: WGS84_A,
            flattening: WGS84_F,
            n: 4.0,
        };
        let atmo = HarrisPriesterAtmosphere {
            model,
            // Altitude tracks |r|; the Sun and satellite direction are fixed so only altitude varies.
            geometry: |_t: f64, r: [f64; 3]| {
                let rm = libm::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                [rm, 0.0, 0.0]
            },
            sun: |_t: f64| SUN,
        };
        let r0 = WGS84_A + 400_000.0;
        let r: [Dual<3>; 3] = [
            Dual::variable(r0, 0),
            Dual::variable(0.0, 1),
            Dual::variable(0.0, 2),
        ];
        let rho = atmo.density(0.0, &r);
        assert!(rho.value() > 0.0 && rho.value().is_finite());
        let dpdx = rho.grad()[0];
        assert!(dpdx.is_finite() && dpdx < 0.0, "dρ/dx = {dpdx:e}");
    }

    /// Unit direction of every scale-height fixture below, off both the equator and the axes:
    /// `0.36² + 0.48² + 0.8² = 1` exactly, so it is a unit vector with no rounding and no zero
    /// component. A fixture along `+x` would leave the `y` and `z` terms of `|pos|` multiplied by a
    /// zero the fixture itself supplied, and the sign of each would be unobservable — measured, not
    /// assumed: `pos[0]·pos[0] − pos[1]·pos[1]` survived the first version of these tests.
    const DIR: [f64; 3] = [0.36, 0.48, 0.8];

    /// Position at radius `r` along [`DIR`].
    fn along(r: f64) -> [f64; 3] {
        [r * DIR[0], r * DIR[1], r * DIR[2]]
    }

    /// The wrapper carrying the fixture closures below. Both capture nothing, so they are plain
    /// function pointers rather than two opaque `impl Fn` parameters.
    type Fixture = HarrisPriesterAtmosphere<fn(f64, [f64; 3]) -> [f64; 3], fn(f64) -> [f64; 3]>;

    /// A radial fixture: altitude tracks `|r|` along a fixed direction, so a `Dual` variable in x is a
    /// pure change of height at a fixed bulge angle.
    fn radial_atmosphere(n: f64) -> Fixture {
        HarrisPriesterAtmosphere {
            model: HarrisPriester {
                equatorial_radius: WGS84_A,
                flattening: WGS84_F,
                n,
            },
            geometry: |_t: f64, r: [f64; 3]| {
                along(libm::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]))
            },
            sun: |_t: f64| SUN,
        }
    }

    /// `∇ρ` at radius `r0`, from the wrapper's `Dual` path. The inertial position is off the axes too:
    /// evaluated at `[r0, 0, 0]` the `y` and `z` terms of `|r|` are multiplied by a zero the fixture
    /// supplied, and ten mutants inside `|r|` and `|r₀|` come back byte-identical.
    fn radial_gradient(atmo: &Fixture, r0: f64) -> [f64; 3] {
        let p = along(r0);
        let r: [Dual<3>; 3] = [
            Dual::variable(p[0], 0),
            Dual::variable(p[1], 1),
            Dual::variable(p[2], 2),
        ];
        atmo.density(0.0, &r).grad()
    }

    /// Radius at which a point along [`DIR`] sits at altitude `alt`. For a fixed direction the
    /// ellipsoid reduction is a constant offset, so one probe determines it.
    fn radius_at_altitude(alt: f64) -> f64 {
        alt + 1.0 - height(along(1.0), WGS84_A, WGS84_F)
    }

    /// The whole `density_pair` — the norm, the 10 km radial offset, the log ratio — is dead in f64:
    /// the lift is `ρ₀·exp((|r₀|−|r|)/H)` evaluated at `|r| = |r₀|`, so it is `exp(0) = 1` whatever `H`
    /// is. `H` is visible only through the derivative, and asserting merely that the derivative is
    /// finite and negative (the test above) is blind to its value. That is why every one of the 29
    /// mutants in `density_pair` survived, and why so did the `−10 000/ln(ρ_up/ρ₀)` line itself.
    ///
    /// The wrapper documents its `H`, so the oracle is the model's own log-slope over the same 10 km,
    /// evaluated by calling [`harris_priester`] directly — the function the Orekit golden sweep pins.
    /// The lift then has to reproduce `∂ρ/∂r = −ρ₀/H` exactly.
    #[test]
    fn the_local_scale_height_is_the_models_own_log_slope() {
        let atmo = radial_atmosphere(4.0);
        for alt in [150_000.0, 400_000.0, 750_000.0] {
            let r0 = radius_at_altitude(alt);
            let rho0 = harris_priester(SUN, along(r0), &atmo.model);
            let rho_up = harris_priester(SUN, along(r0 + 10_000.0), &atmo.model);
            let h = -10_000.0 / libm::log(rho_up / rho0);
            let slope = -rho0 / h; // ∂ρ/∂|r|; the chain rule gives ∂ρ/∂rᵢ = slope · rᵢ/|r|
            let got = radial_gradient(&atmo, r0);
            let mut rel = 0.0_f64;
            for i in 0..3 {
                rel = nan_max(rel, ((got[i] - slope * DIR[i]) / (slope * DIR[i])).abs());
            }
            println!("HP {alt:.0} m: H = {h:.1} m, ∇ρ {got:?}, ∂ρ/∂|r| {slope:e} (rel {rel:e})");
            assert!(
                rel < 1.0e-12,
                "at {alt:.0} m the lift's scale height is not the model's: {got:?} vs {slope:e}·DIR"
            );
            // Not the fallback: if it were, the assertion above would be measuring the constant.
            assert!(
                ((h - 50_000.0) / 50_000.0).abs() > 0.05,
                "the fixture landed on the 50 km fallback, so it proves nothing: H = {h:.1}"
            );
        }
    }

    /// Ten kilometres above 995 km is past the table ceiling, so `ρ_up` is exactly zero, the log ratio
    /// is `−∞` and the wrapper must take its documented 50 km fallback. `ρ₀` is still positive there,
    /// which is what makes the fallback observable at all: above the ceiling `ρ₀` is zero too and the
    /// whole product vanishes whatever `H` is.
    #[test]
    fn just_below_the_ceiling_the_lift_falls_back_to_fifty_kilometres() {
        let atmo = radial_atmosphere(4.0);
        let r0 = radius_at_altitude(995_000.0);
        let rho0 = harris_priester(SUN, along(r0), &atmo.model);
        assert!(rho0 > 0.0, "fixture is above the ceiling, not below it");
        assert_eq!(
            harris_priester(SUN, along(r0 + 10_000.0), &atmo.model),
            0.0,
            "fixture does not reach past the ceiling"
        );
        let got = radial_gradient(&atmo, r0);
        let slope = -rho0 / 50_000.0;
        println!("HP 995 km fallback: ∇ρ {got:?}, ∂ρ/∂|r| {slope:e}");
        for i in 0..3 {
            let want = slope * DIR[i];
            assert!(
                ((got[i] - want) / want).abs() < 1.0e-14,
                "the ceiling fallback is not 50 km: {:e} vs {want:e}",
                got[i]
            );
        }
    }
}
