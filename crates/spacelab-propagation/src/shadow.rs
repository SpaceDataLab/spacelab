//! Conical umbra/penumbra shadow geometry for radiation forces — the lighting ratio `ν ∈ [0, 1]`,
//! the fraction of the solar disc a spherical occulting body leaves visible.
//!
//! Seen from the satellite, the Sun and the occulter are discs of apparent radii `r_S` and `r_E`
//! whose centres are `d` apart, all three angles. `ν` is one minus the area `A` the two discs share,
//! taken as plane, over the Sun's `π r_S²`: exactly `0` in the umbra, `1` in full sun and the lens in
//! between. It is `no_std` so the numerical force and the DSST mean-element path share one
//! definition. `ν` is used as an f64 value gate — the penumbra boundary is a measure-zero set the STM
//! is not defined across, so it stays out of the Jacobian.
//!
//! # The lens without cancellation
//!
//! [`lens_area`] writes `A` as two circular segments on the common chord, `r²(x − sin x)/2` each, `x`
//! the angle the chord subtends at that disc's centre: `x/2 = atan2(4K, d² + r² − r'²)`, where
//! `4K = √((r + r' − d)(d − r + r')(d + r − r')(d + r + r'))` is four times Heron's area of the
//! triangle of the two centres and a crossing point. The product is taken in factored form and
//! `x − sin x` by its Taylor series below `x = 2`, its leading terms in double-double, so neither
//! tangency cancels, and the plateaus are decided by comparing `d` with `r_E + r_S` and
//! `|r_E − r_S|` — no angular margin.
//!
//! Orekit's arrangement — `acos` of each chord distance, less the chord's triangle — is the same area
//! (`derivations/shadow_lens.mac`) and loses digits at both tangencies and in between: on the grid
//! of `tests/shadow_lens.rs`, for a 400 km orbit, it is 6.5e-7 off in `ν` 1.1e-9 rad inside either
//! tangency, where it returns `ν < 0`, and 1e-10 to 9e-10 off across the penumbra, because there the
//! occulter's segment is the difference of two terms 1e5 times its size.
//!
//! # Checked against
//!
//! `tests/shadow_lens.rs`: on 184 exact binary64 inputs — seven radius pairs from a 400 km orbit's
//! Earth (`r_E/r_S = 264`) to an occulter inside the Sun's disc, both tangencies to the last double,
//! `d → 0`, and 13 points off that grid — [`lens_area`] against `derivations/shadow_lens.gp`'s
//! 200-digit values, each inside an Arb ball (`derivations/shadow_lens_arb.c`). The bound is
//! `18 + 2κ` ulp of `A`, `κ = 4K/A` its condition number in `d`, derived from every rounding the code
//! performs (`derivations/shadow_lens_budget.mac`, `shadow_lens_budget_arb.c`);
//! `derivations/shadow_lens.mac` shows the segment form is the textbook area.

use libm::{asin, atan2, fabs, sin, sqrt};

/// Sun radius (m) — Orekit `Constants.SUN_RADIUS`.
pub const SUN_RADIUS: f64 = 6.957e8;

fn norm(v: [f64; 3]) -> f64 {
    sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The angle between two vectors, `atan2(|u × v|, u·v)`: accurate over the whole range, where
/// `acos` of the normalised dot product loses digits near 0 and π and `asin` of the cross product
/// near π/2.
fn angle(u: [f64; 3], v: [f64; 3]) -> f64 {
    atan2(norm(cross(u, v)), dot(u, v))
}

/// `a · b` as `hi + lo` exactly: Dekker's product on Veltkamp's halves (Dekker 1971), so no fused
/// multiply-add is needed. Exact while neither the product nor its error term leaves the normal
/// range.
fn two_product(a: f64, b: f64) -> (f64, f64) {
    const SPLIT: f64 = 134_217_729.0; // 2^27 + 1
    let halves = |v: f64| {
        let c = SPLIT * v;
        let hi = c - (c - v);
        (hi, v - hi)
    };
    let p = a * b;
    let (ah, al) = halves(a);
    let (bh, bl) = halves(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

/// `(hi + lo) / k` as a pair whose sum is within a few `u²` of it, for a small integer `k`: the
/// quotient's remainder `hi − k·q` is exact by Dekker's product and Sterbenz.
fn divide(hi: f64, lo: f64, k: f64) -> (f64, f64) {
    let q = hi / k;
    let (p, e) = two_product(q, k);
    (q, (((hi - p) - e) + lo) / k)
}

/// `x − sin x` for `x ∈ [0, 2π]`. Below `x = 2` it is the alternating Taylor series
/// `x³/3! − x⁵/5! + x⁷/7! Q`, `Q = 1 − x²/(8·9) (1 − x²/(10·11) (… (1 − x²/(24·25))))` to `x²⁵/25!`:
/// the first two terms as double-double pairs, their difference by Knuth's two-sum, and the rest,
/// at most `0.024` of the result, in plain doubles. The terms fall by at least `x²/20 < 1/5` each, so
/// the first one left out, `x²⁷/27!`, bounds the truncation: under `1.2e-20` of the sum. Everything
/// before the last rounding is within `0.245u` of `x − sin x`, so the result is within `1.245u`
/// relative and `0.745` ulp for `x ≥ 2⁻¹⁸⁰` (`derivations/x_minus_sin_roundoff.mac`). Above `x = 2`,
/// `|sin x|` is at most `0.84` of `x − sin x`, so the subtraction keeps its digits: within
/// `1 + 2|sin x|/(x − sin x) ≤ 2.67` u, libm's ulp of `sin x` and one rounding.
fn x_minus_sin(x: f64) -> f64 {
    if x < 2.0 {
        let (x2, x2l) = two_product(x, x);
        let (x3, e3) = two_product(x, x2);
        let x3l = e3 + x * x2l;
        let (x5, e5) = two_product(x3, x2);
        let x5l = e5 + (x3 * x2l + x3l * x2);
        let (t3, t3l) = divide(x3, x3l, 6.0);
        let (t5, t5l) = divide(x5, x5l, 120.0);
        let mut q = 1.0;
        for n in [24.0, 22.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0] {
            q = 1.0 - x2 * q / (n * (n + 1.0));
        }
        let tail = x5 * x2 * q / 5040.0;
        let head = t3 - t5;
        let back = head - t3;
        let head_lo = (t3 - (head - back)) - (t5 + back);
        head + (((head_lo + t3l) - t5l) + tail)
    } else {
        x - sin(x)
    }
}

/// The area two discs share in the plane: radii `alpha_central` (the occulter) and `alpha_sun`, their
/// centres `separation` apart — for [`masking_ratio`], apparent radii and separation in radians.
///
/// `0` when the discs are apart, `π·min(r)²` when one lies inside the other, and in between the two
/// circular segments on the common chord, `r²(x − sin x)/2` each, with `x/2 = atan2(4K, d² + r² − r'²)`
/// and `4K` Heron's product in factored form (module docs). Within `(18 + 2κ)u` relative of the exact
/// area of the given doubles, so within `18 + 2κ` ulp, `κ = 4K/A`: to first order in the roundings,
/// over the whole penumbra (`derivations/shadow_lens_budget.mac`).
#[must_use]
pub fn lens_area(separation: f64, alpha_central: f64, alpha_sun: f64) -> f64 {
    use core::f64::consts::PI;
    let d = separation;
    let sum = alpha_central + alpha_sun;
    if d >= sum {
        return 0.0;
    }
    let diff = alpha_central - alpha_sun;
    if d <= fabs(diff) {
        let inner = alpha_central.min(alpha_sun);
        return PI * inner * inner;
    }
    let heron = sqrt((sum - d) * (d - diff) * (d + diff) * (d + sum));
    // r_E² − r_S², factored: the two chord-angle denominators are d² ∓ it.
    let squares = diff * sum;
    let d2 = d * d;
    let sun = alpha_sun * alpha_sun * x_minus_sin(2.0 * atan2(heron, d2 - squares));
    let central = alpha_central * alpha_central * x_minus_sin(2.0 * atan2(heron, d2 + squares));
    0.5 * (sun + central)
}

/// Fraction of the occulted (Sun) disc still visible, given the apparent angular `separation` of the
/// two disc centres, the occulting body's apparent (`alpha_central`) and the Sun's apparent
/// (`alpha_sun`) radii: `1 − A/(π α_sun²)` with `A` the [`lens_area`]. `0` in full umbra, `1` in full
/// sun, the lens fraction in penumbra.
#[must_use]
pub fn masking_ratio(separation: f64, alpha_central: f64, alpha_sun: f64) -> f64 {
    use core::f64::consts::PI;
    let disc = PI * alpha_sun * alpha_sun;
    let visible = 1.0 - lens_area(separation, alpha_central, alpha_sun) / disc;
    // Rounding can carry a lens a hair past the whole disc next to the umbra; a NaN passes through.
    if visible < 0.0 { 0.0 } else { visible }
}

/// Conical lighting ratio `λ ∈ [0, 1]` (Sun visibility) at geocentric position `pos` (m), for the Sun
/// at `sun_pos` (m) and a single spherical occulting body of radius `occulting_radius` (m, e.g. the
/// Earth): the [`masking_ratio`] of the apparent discs. A position inside the occulting sphere, where
/// its apparent radius has no value, sees no Sun: `0`.
#[must_use]
pub fn lighting_ratio(pos: [f64; 3], sun_pos: [f64; 3], occulting_radius: f64) -> f64 {
    if norm(sun_pos) < 2.0 * SUN_RADIUS {
        return 1.0; // degenerate (Sun-centred frame); kept for faithfulness.
    }
    let distance = norm(pos);
    if distance < occulting_radius {
        return 0.0;
    }
    let sun_sat = [
        pos[0] - sun_pos[0],
        pos[1] - sun_pos[1],
        pos[2] - sun_pos[2],
    ];
    let separation = angle(sun_sat, pos);
    let occulted_apparent_radius = asin(SUN_RADIUS / norm(sun_sat));
    let limb_radius = asin(occulting_radius / distance);
    masking_ratio(separation, limb_radius, occulted_apparent_radius)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn masking_ratio_umbra_penumbra_sun() {
        // Earth apparent radius 0.1 rad, Sun 0.01 rad.
        assert!(masking_ratio(0.0, 0.1, 0.01) < 1e-12); // Sun fully behind Earth
        assert!((masking_ratio(0.2, 0.1, 0.01) - 1.0).abs() < 1e-12); // well clear → full sun
        let p = masking_ratio(0.1, 0.1, 0.01); // Sun centre on Earth's limb → partial
        assert!(p > 0.0 && p < 1.0, "penumbra {p}");
        // Monotone: wider separation → more Sun visible.
        assert!(masking_ratio(0.105, 0.1, 0.01) > p);
    }

    /// `x − sin x` from `derivations/shadow_lens.gp` at 200 digits, each value inside an Arb ball, as
    /// `(x, hi, lo)`: both sides of the series' switch at 2, its far end, and the exact double next to
    /// `2π`.
    #[expect(
        clippy::unreadable_literal,
        clippy::approx_constant,
        reason = "printed by the script and checked verbatim; the doubles nearest π and 2π are inputs"
    )]
    #[rustfmt::skip]
    const X_MINUS_SIN: [(f64, f64, f64); 13] = [
    (9.313225746154785e-10, 1.3463226115771934e-28, 7.467753066131004e-45),
    (1e-8, 1.6666666666666668e-25, -6.143435497173705e-43),
    (1e-3, 1.6666665833333355e-10, -5.6979554413661525e-27),
    (5e-1, 2.0574461395796998e-2, 1.6345229086023983e-18),
    (1e0, 1.585290151921035e-1, -1.776845092935536e-18),
    (1.5e0, 5.025050133959456e-1, 1.4558643538840918e-17),
    (1.9999999999999998e0, 1.090702573174318e0, -7.838225335409553e-17),
    (2e0, 1.0907025731743183e0, 1.4020906557816256e-17),
    (2.0000000000000004e0, 1.090702573174319e0, -2.321737854339134e-17),
    (2.5e0, 1.9015278558960436e0, -5.580826912169191e-17),
    (3.141592653589793e0, 3.141592653589793e0, -1.2246467991473532e-16),
    (6e0, 6.279415498198926e0, 1.2659979684764697e-17),
    (6.283185307179586e0, 6.283185307179586e0, 2.4492935982947064e-16),
    ];

    /// The series branch densely, from the same script: every `k/32` and every tenth below 2, the
    /// point where the old term-by-term sum was 4.44 ulp off (1.858598579272284), and the worst
    /// points of a 2 000 006-point scan of the shipped series.
    #[expect(clippy::unreadable_literal, reason = "printed by the script and checked verbatim")]
    #[rustfmt::skip]
    const X_MINUS_SIN_SERIES: [(f64, f64, f64); 79] = [
    (3.125e-2, 5.08601467392126e-6, 5.774541248819496e-24),
    (6.25e-2, 4.0682157619801415e-5, 6.041675973557332e-22),
    (9.375e-2, 1.37268764487107e-4, 8.097322675576633e-22),
    (1e-1, 1.665833531718477e-4, 9.961062288767823e-21),
    (1.25e-1, 3.2526661477231004e-4, -1.3983696530037811e-21),
    (1.5625e-1, 6.350072264439588e-4, 4.4044420388485705e-21),
    (1.875e-1, 1.0967032377301154e-3, 3.586508933995256e-20),
    (2e-1, 1.3306692049387847e-3, 3.3555772988158105e-20),
    (2.1875e-1, 1.7404189049898431e-3, 1.0563152048487e-19),
    (2.5e-1, 2.5960407454770702e-3, 1.5845018300563033e-19),
    (2.8125e-1, 3.693248353663674e-3, 1.0684536597044932e-19),
    (3e-1, 4.4797933386604245e-3, -1.0076077903606229e-19),
    (3.125e-1, 5.0614854196191494e-3, -1.1004366442765296e-19),
    (3.4375e-1, 6.729930977746924e-3, 9.606099507362659e-20),
    (3.75e-1, 8.727470913952439e-3, -4.69526293754318e-19),
    (4e-1, 1.058165769134951e-2, -6.102971666250016e-20),
    (4.0625e-1, 1.1082669759065763e-2, 5.315296356693493e-19),
    (4.375e-1, 1.3823742796061989e-2, 7.666018129902178e-19),
    (4.6875e-1, 1.6978528508316222e-2, 1.2951800381916742e-18),
    (5.3125e-1, 2.463854518574263e-2, 1.4691116685991519e-18),
    (5.625e-1, 2.9197326463979826e-2, 7.485231289837725e-19),
    (5.9375e-1, 3.4276868752633124e-2, -1.8778673370728243e-18),
    (6e-1, 3.535752660496464e-2, -2.1941104937027437e-18),
    (6.25e-1, 3.9902727059537846e-2, -6.271787700960239e-19),
    (6.5625e-1, 4.609992292420863e-2, 9.204820997754244e-19),
    (6.875e-1, 5.28929199847307e-2, -1.2588712691117479e-19),
    (7e-1, 5.578231276230894e-2, -2.8740567927338755e-18),
    (7.1875e-1, 6.0305600089432455e-2, 3.0419171807705746e-18),
    (7.5e-1, 6.836123997666584e-2, -2.4713097085356585e-18),
    (7.8125e-1, 7.708248854546633e-2, -2.2237933649608737e-18),
    (8e-1, 8.264390910047725e-2, -1.1851419720716684e-18),
    (8.125e-1, 8.649134473928745e-2, 1.8584303455814125e-18),
    (8.4375e-1, 9.660913606440577e-2, -1.6194122646532092e-18),
    (8.75e-1, 1.0745649776397297e-1, -5.8985321879771185e-18),
    (9e-1, 1.1667309037251662e-1, -1.908188809964743e-18),
    (9.0625e-1, 1.1905335266805106e-1, 4.500814036448646e-19),
    (9.375e-1, 1.3141889173930701e-1, -9.581959135080334e-18),
    (9.6875e-1, 1.4457155533336327e-1, -1.2632096669432433e-17),
    (1.03125e0, 1.7330815718751652e-1, 4.097477302943774e-18),
    (1.0625e0, 1.8892506483292887e-1, 1.1342141201441087e-17),
    (1.09375e0, 2.0539500357772697e-1, 2.526759081024206e-18),
    (1.1e0, 2.087926399385647e-1, 8.303729613773933e-18),
    (1.125e0, 2.2273240590090485e-1, -8.060268334979814e-18),
    (1.15625e0, 2.409508572179933e-1, 4.715837465848465e-18),
    (1.1875e0, 2.6006308261513233e-1, -6.645726005605572e-18),
    (1.2e0, 2.679609140327736e-1, 1.036460881230838e-17),
    (1.21875e0, 2.800809344232402e-1, 6.754915874182965e-18),
    (1.25e0, 3.010153806444138e-1, -1.3508965656504773e-17),
    (1.28125e0, 3.2287649424186027e-1, -1.2785048983819597e-17),
    (1.3e0, 3.364418145828071e-1, -1.8247650480909386e-17),
    (1.3125e0, 3.4567344330381977e-1, -1.771640581949128e-18),
    (1.34375e0, 3.6941448209108274e-1, 6.0506280423808274e-18),
    (1.375e0, 3.941069429768443e-1, 1.613707158139294e-17),
    (1.4e0, 4.1455027001153977e-1, -2.200402298371803e-17),
    (1.40625e0, 4.197572292867663e-1, -1.9449459906232266e-17),
    (1.4375e0, 4.463708090462383e-1, 4.121690732375912e-18),
    (1.46875e0, 4.7395220964094403e-1, 1.8275853850004697e-17),
    (1.53125e0, 5.32031854077604e-1, -3.726031261620716e-17),
    (1.5625e0, 5.625344143217511e-1, 1.633274480620419e-17),
    (1.59375e0, 5.940134239906244e-1, -2.7111783461603818e-17),
    (1.6e0, 6.00426396958495e-1, -5.1870146867022216e-17),
    (1.625e0, 6.264686594601684e-1, 2.958300233854839e-17),
    (1.65625e0, 6.598989438384004e-1, -2.6749645464721284e-18),
    (1.6875e0, 6.943021481146251e-1, -4.0503049291509105e-17),
    (1.7e0, 7.083351895475314e-1, -5.077812604198573e-17),
    (1.71875e0, 7.296751932383774e-1, -4.271139167030821e-17),
    (1.75e0, 7.660140531260631e-1, 2.4308897094982022e-17),
    (1.78125e0, 8.033137585923614e-1, -4.431923170524409e-17),
    (1.8e0, 8.261523691218049e-1, -2.90678192930995e-17),
    (1.8125e0, 8.415684022025495e-1, 1.4404590742971085e-17),
    (1.84375e0, 8.807711440410128e-1, 1.814986636328818e-17),
    (1.858598579272284e0, 8.99728567674338e-1, -4.9202777212515743e-17),
    (1.875e0, 9.209142183903062e-1, 1.7763371808564367e-18),
    (1.9e0, 9.536999123125854e-1, -4.6531737366356815e-17),
    (1.90625e0, 9.619889413142455e-1, 2.8020804737642804e-17),
    (1.9100976999889905e0, 9.671102815271663e-1, 4.763364268255673e-17),
    (1.9375e0, 1.0039857191376238e0, 1.8047010573845976e-17),
    (1.9376388993070526e0, 1.0041744284374043e0, -1.0472298578422326e-16),
    (1.96875e0, 1.0468940578142722e0, 1.223145636473661e-17),
    ];

    /// The spacing of doubles at `x > 0`.
    fn ulp(x: f64) -> f64 {
        f64::from_bits(x.to_bits() + 1) - x
    }

    /// Each branch within its proved bound: the series within 0.745 ulp
    /// (`derivations/x_minus_sin_roundoff.mac`), the subtraction within `1 + 2|sin x|/(x − sin x)`
    /// ulp, one rounding and libm's one ulp of `sin x`.
    #[test]
    fn x_minus_sin_is_within_its_proved_bound_of_the_reference() {
        let (mut series, mut direct) = (0.0_f64, 0.0_f64);
        for &(x, hi, lo) in X_MINUS_SIN.iter().chain(&X_MINUS_SIN_SERIES) {
            let err = fabs((x_minus_sin(x) - hi) - lo) / ulp(hi);
            let bound = if x < 2.0 {
                0.75
            } else {
                1.0 + 2.0 * fabs(x - hi) / hi
            };
            assert!(err <= bound, "x = {x:e}: {err:.3} ulp against {bound:.3}");
            if x < 2.0 {
                series = series.max(err);
            } else {
                direct = direct.max(err / bound);
            }
        }
        std::println!(
            "x − sin x: series worst {series:.3} ulp; direct worst {direct:.3} of its bound"
        );
    }

    #[test]
    fn lighting_ratio_full_sun_and_umbra() {
        let sun = [1.496e11, 0.0, 0.0];
        let ae = 6.378_137e6;
        // Sun-side satellite → fully lit.
        assert!((lighting_ratio([7.0e6, 0.0, 0.0], sun, ae) - 1.0).abs() < 1e-12);
        // Directly behind the Earth, just above the surface → deep umbra.
        assert!(lighting_ratio([-6.6e6, 0.0, 0.0], sun, ae) < 1e-9);
        // Far off the shadow axis → fully lit again.
        assert!((lighting_ratio([-6.6e6, 1.0e8, 0.0], sun, ae) - 1.0).abs() < 1e-12);
    }

    const EARTH_RADIUS: f64 = 6.378_137e6;

    /// An orthogonal pair with no zero component, both exact in binary64: `|p| = |q| = 7` and
    /// `p·q = 12 + 6 − 18 = 0`. Everything below is built in their plane, so no component of any
    /// input vanishes and no cross-product row is dead.
    const P: [f64; 3] = [2.0, 3.0, 6.0];
    const Q: [f64; 3] = [6.0, 2.0, -3.0];

    /// A vector of length `len` at exactly `theta` from [`P`], in the `P`–`Q` plane.
    fn at_angle(theta: f64, len: f64) -> [f64; 3] {
        let (c, s) = (libm::cos(theta), libm::sin(theta));
        let k = len / 7.0;
        [
            (c * P[0] + s * Q[0]) * k,
            (c * P[1] + s * Q[1]) * k,
            (c * P[2] + s * Q[2]) * k,
        ]
    }

    /// `norm`, `dot` and `cross` against their definitions on integers, where the expected value is
    /// checkable by inspection. Every call site in this module feeds them geometry symmetric about
    /// the shadow axis, and a zero component makes most of their arithmetic unobservable.
    #[test]
    fn the_vector_helpers_match_their_definitions() {
        assert_eq!(norm([2.0, 3.0, 6.0]), 7.0); // 4 + 9 + 36 = 49
        assert_eq!(dot([1.0, 2.0, 3.0], [4.0, 5.0, 7.0]), 35.0); // 4 + 10 + 21
        assert_eq!(dot([1.0, -2.0, 3.0], [4.0, 5.0, -7.0]), -27.0); // 4 − 10 − 21
        assert_eq!(cross([1.0, 2.0, 3.0], [4.0, 5.0, 7.0]), [-1.0, 5.0, -3.0]);
    }

    /// `angle` against the angle its inputs were *built* from, from 1e-8 to π − 1e-8, where an
    /// `acos` of the dot product would lose half its digits at either end and an `asin` of the cross
    /// product all of them at π/2. `|u| = 7` and `|v| = 3` keep `norm_product` off 1.
    #[test]
    fn angle_recovers_the_angle_its_inputs_were_built_from() {
        use core::f64::consts::{FRAC_PI_2, PI};
        for &theta in &[
            1.0e-8,
            1.0e-4,
            0.3,
            1.0,
            FRAC_PI_2 - 1.0e-9,
            FRAC_PI_2,
            FRAC_PI_2 + 1.0e-9,
            2.0,
            PI - 1.0e-4,
            PI - 1.0e-8,
        ] {
            let got = angle(P, at_angle(theta, 3.0));
            assert!((got - theta).abs() < 1e-15, "theta {theta}: got {got}");
        }
    }

    /// Both plateaus are exact, not approximate, and so is their edge: at external tangency the lens
    /// is empty, and just inside internal tangency the Sun's disc is wholly covered, so `0` and `1`
    /// are asserted as equalities.
    #[test]
    fn masking_ratio_is_exact_on_both_plateaus() {
        assert_eq!(masking_ratio(0.05, 0.1, 0.01), 0.0); // Sun disc well inside the Earth disc
        assert_eq!(masking_ratio(0.089, 0.1, 0.01), 0.0); // just inside internal tangency
        assert_eq!(masking_ratio(0.11, 0.1, 0.01), 1.0); // external tangency
        assert_eq!(masking_ratio(0.2, 0.1, 0.01), 1.0);
    }

    /// An occulter of zero radius blocks nothing, wherever it stands on the Sun's disc: the lens is
    /// the disc inside the other, `π·0²`.
    #[test]
    fn a_zero_radius_occulter_blocks_nothing() {
        assert_eq!(masking_ratio(1.0e-10, 0.0, 2.0e-10), 1.0);
        assert_eq!(masking_ratio(0.0, 0.0, 2.0e-10), 1.0);
        assert_eq!(lens_area(1.0e-10, 0.0, 2.0e-10), 0.0);
    }

    /// Inside the occulting sphere `asin(R/|r|)` has no value, and the answer is no Sun, even on the
    /// side that faces it; on the surface itself the geometry is evaluated, the body a hemisphere.
    #[test]
    fn a_position_inside_the_occulter_sees_no_sun() {
        let sun = [1.496e11, 0.0, 0.0];
        assert_eq!(lighting_ratio([1.0e3, 0.0, 0.0], sun, EARTH_RADIUS), 0.0);
        assert_eq!(
            lighting_ratio([EARTH_RADIUS - 1.0, 0.0, 0.0], sun, EARTH_RADIUS),
            0.0
        );
        assert_eq!(
            lighting_ratio([EARTH_RADIUS, 0.0, 0.0], sun, EARTH_RADIUS),
            1.0
        );
        assert_eq!(
            lighting_ratio([-EARTH_RADIUS, 0.0, 0.0], sun, EARTH_RADIUS),
            0.0
        );
    }

    /// The degenerate-Sun guard, whose threshold no other case reaches. Below it the geometry is
    /// meaningless and the answer is full sun even for a satellite the geometry would put in deep
    /// umbra; at the threshold itself, and above, the geometry is evaluated.
    #[test]
    fn lighting_ratio_short_circuits_a_sun_centred_frame() {
        let pos = [-6.6e6, 0.0, 0.0];
        assert_eq!(lighting_ratio(pos, [1.0e9, 0.0, 0.0], EARTH_RADIUS), 1.0);
        let at_threshold = [2.0 * SUN_RADIUS, 0.0, 0.0];
        assert!(lighting_ratio(pos, at_threshold, EARTH_RADIUS) < 1e-12);
        assert!(lighting_ratio(pos, [1.6e9, 0.0, 0.0], EARTH_RADIUS) < 1e-12);
    }

    /// A penumbra crossing with nothing on an axis: `lighting_ratio` from the vectors against
    /// `masking_ratio` at the separation and apparent radii the geometry was *constructed* from.
    /// Every case above lies on the shadow axis, where `λ` is pinned to a plateau and the entire
    /// vector path — the `pos − sun_pos` difference, `angle`, `cross` — can be wrong without moving
    /// it. The two differ by what building the vectors and measuring the angle back cost, at most
    /// `∂ν/∂d ≤ 2/(π α_s) = 137 rad⁻¹` times a few ulp of the 1.14 rad separation.
    #[test]
    fn lighting_ratio_crosses_the_penumbra_off_axis() {
        let d_sun = 1.496e11;
        let pos = at_angle(0.0, 7.0e6); // [2, 3, 6]·10⁶ m, |pos| = 7000 km exactly
        let alpha_c = asin(EARTH_RADIUS / 7.0e6);
        let alpha_s = asin(SUN_RADIUS / d_sun);
        let mut worst = 0.0_f64;
        for k in -4_i32..=4 {
            let separation = alpha_c + f64::from(k) * 0.4 * alpha_s;
            let sun_sat = at_angle(separation, d_sun);
            let sun_pos = [
                pos[0] - sun_sat[0],
                pos[1] - sun_sat[1],
                pos[2] - sun_sat[2],
            ];
            let got = lighting_ratio(pos, sun_pos, EARTH_RADIUS);
            let want = masking_ratio(separation, alpha_c, alpha_s);
            if k.abs() <= 2 {
                assert!(got > 0.0 && got < 1.0, "k={k}: {got} is not a penumbra");
            }
            worst = worst.max((got - want).abs());
        }
        std::println!("lighting_ratio vs masking_ratio off axis: worst {worst:.2e}");
        assert!(worst < 1e-12, "{worst:.3e}");
    }
}
