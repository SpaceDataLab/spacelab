//! The Galileo NeQuick-G driver: effective ionisation level, electron density and STEC.
//!
//! Ported from Orekit's `NeQuickGalileo` (Galileo specialisation) together with the
//! electron-density core of `NeQuickModel`, then brought into line with the Galileo ICD
//! (issue 1.2) where Orekit departs from it. The three broadcast coefficients map the MODIP
//! at the receiver to the effective ionisation level Az; the model then integrates the
//! electron density along the receiver-to-satellite ray with the ICD's adaptive Kronrod
//! G7-K15 quadrature (Annex F.2.6), the method that produced the Annex E verification data.
//!
//! Gate: `derivations/kronrod_g7k15.mac` builds the G7 rule from P7 and the K15 extension from
//! the Stieltjes polynomial E8, shows that they integrate polynomials to degrees 13 and 22 and no
//! further, that the centre G7 weight is 512/1225, and that every node and weight the ICD prints
//! is the defined value rounded to its 33 digits; `derivations/kronrod_g7k15.gp` repeats all of
//! it at 80 digits. The module tests `the_kronrod_nodes_and_weights_are_their_defining_values`
//! (G2) and `the_all_zero_alpha_sentinel_is_the_only_thing_that_yields_the_default_az` (G2, Eq. 17
//! and 18 with the section 3.3 clamp) hold the code to them.

use crate::ccir::{self};
use crate::fourier::{EpochSeries, FourierTimeSeries};
use crate::math::{clip_exp, join};
use crate::model::{self, Variant};
use crate::modip::ModipGrid;
use crate::parameters::{AmplitudeRules, NeQuickParameters};
use crate::ray::Ray;
use crate::types::{DateTimeComponents, GeodeticPoint};

/// Heights, m, at which ICD Eq. 188 to 190 split the slant path.
const SPLIT_HEIGHTS: [f64; 2] = [1_000_000.0, 2_000_000.0];

/// Recursion limit of the adaptive quadrature, the JRC reference implementation's value.
const MAX_RECURSION: u32 = 50;

/// Kronrod K15 abscissae on [-1, 1] (ICD Annex F.2.6); the odd-indexed ones are the G7 nodes.
#[expect(
    clippy::excessive_precision,
    reason = "the ICD Annex F.2.6 values, kept at its 33 digits"
)]
const K15_NODES: [f64; 15] = [
    -0.991455371120812639206854697526329,
    -0.949107912342758524526189684047851,
    -0.864864423359769072789712788640926,
    -0.741531185599394439863864773280788,
    -0.586087235467691130294144838258730,
    -0.405845151377397166906606412076961,
    -0.207784955007898467600689403773245,
    0.0,
    0.207784955007898467600689403773245,
    0.405845151377397166906606412076961,
    0.586087235467691130294144838258730,
    0.741531185599394439863864773280788,
    0.864864423359769072789712788640926,
    0.949107912342758524526189684047851,
    0.991455371120812639206854697526329,
];

/// Kronrod K15 weights (ICD Annex F.2.6).
#[expect(
    clippy::excessive_precision,
    reason = "the ICD Annex F.2.6 values, kept at its 33 digits"
)]
const K15_WEIGHTS: [f64; 15] = [
    0.022935322010529224963732008058970,
    0.063092092629978553290700663189204,
    0.104790010322250183839876322541518,
    0.140653259715525918745189590510238,
    0.169004726639267902826583426598550,
    0.190350578064785409913256402421014,
    0.204432940075298892414161999234649,
    0.209482141084727828012999174891714,
    0.204432940075298892414161999234649,
    0.190350578064785409913256402421014,
    0.169004726639267902826583426598550,
    0.140653259715525918745189590510238,
    0.104790010322250183839876322541518,
    0.063092092629978553290700663189204,
    0.022935322010529224963732008058970,
];

/// Gauss G7 weights for the odd-indexed K15 nodes (ICD Annex F.2.6).
#[expect(
    clippy::excessive_precision,
    reason = "the ICD Annex F.2.6 values, kept at its 33 digits"
)]
const G7_WEIGHTS: [f64; 7] = [
    0.129484966168869693270611432679082,
    0.279705391489276667901467771423780,
    0.381830050505118944950369775488975,
    0.417959183673469387755102040816327,
    0.381830050505118944950369775488975,
    0.279705391489276667901467771423780,
    0.129484966168869693270611432679082,
];

/// NeQuick-G ionospheric model driven by the three Galileo broadcast coefficients.
///
/// Construct it once with the effective-ionisation coefficients `α = [a0, a1, a2]`
/// broadcast in the Galileo navigation message, then call [`NeQuickGalileo::stec`] for
/// each receiver/satellite geometry and epoch.
pub struct NeQuickGalileo {
    /// Broadcast effective-ionisation coefficients.
    alpha: [f64; 3],
    /// Galileo MODIP grid.
    modip: ModipGrid,
}

impl NeQuickGalileo {
    /// Build the model from the three broadcast effective-ionisation coefficients.
    #[must_use]
    pub fn new(alpha: [f64; 3]) -> Self {
        Self {
            alpha,
            modip: ModipGrid::galileo(),
        }
    }

    /// The broadcast effective-ionisation coefficients this model was built with.
    #[must_use]
    pub fn alpha(&self) -> [f64; 3] {
        self.alpha
    }

    /// Slant Total Electron Content along the receiver-to-satellite ray, in TECU.
    ///
    /// `receiver` and `satellite` are geodetic points (latitude/longitude in radians,
    /// altitude in metres); `date` supplies the calendar month and universal time.
    #[must_use]
    pub fn stec(
        &self,
        date: DateTimeComponents,
        receiver: GeodeticPoint,
        satellite: GeodeticPoint,
    ) -> f64 {
        let ray = Ray::new(receiver, satellite);
        let month = date.month();
        let hours = date.hours();
        // ICD section 2.5.4.4: Az is fixed by the MODIP at the receiver and is not updated along
        // the ray, so the whole CCIR collapse is the same at every node.
        let receiver_modip = self.modip.compute(receiver.latitude, receiver.longitude);
        let epoch = EpochSeries::new(hours, ccir::load(month));
        let fourier =
            FourierTimeSeries::at(self.effective_ionization_level(receiver_modip), &epoch);
        let density = |s: f64| {
            let gp = ray.point_at(s);
            // The local MODIP still drives the spatial (Legendre) terms of foF2 and M(3000)F2.
            let modip = self.modip.compute(gp.latitude, gp.longitude);
            model::electron_density(&GalileoVariant, &fourier, month, hours, gp, modip)
        };

        // A receiver below the ellipsoid integrates from zero height, as the reference does.
        let mut lower_h = ray.rec_h().max(0.0);
        let mut lower = ray.distance_at(lower_h);
        let mut tec = 0.0;
        for split_h in splits(ray.rec_h(), ray.sat_h()) {
            let upper = ray.distance_at(split_h);
            tec += kronrod(&density, lower, upper, segment_tolerance(lower_h), 0);
            (lower_h, lower) = (split_h, upper);
        }
        tec += kronrod(&density, lower, ray.s2(), segment_tolerance(lower_h), 0);

        tec * 1.0e-16
    }

    /// Effective ionisation level Az from a MODIP (degrees), Eq. 17/18.
    #[allow(clippy::float_cmp)] // exact all-zero sentinel is the reference's special case (Eq. 17)
    fn effective_ionization_level(&self, modip: f64) -> f64 {
        let [a0, a1, a2] = self.alpha;
        if a0 == 0.0 && a1 == 0.0 && a2 == 0.0 {
            63.7
        } else {
            (a0 + modip * (a1 + modip * a2)).clamp(0.0, 400.0)
        }
    }
}

/// The split heights, m, that fall strictly inside a path from `rec_h` to `sat_h` (Eq. 188, 189).
fn splits(rec_h: f64, sat_h: f64) -> impl Iterator<Item = f64> {
    SPLIT_HEIGHTS
        .into_iter()
        .filter(move |&h| rec_h < h && h < sat_h)
}

/// Integration accuracy of a segment starting at `lower_h`, m: 1e-3 below 1000 km, 1e-2 above
/// (Eq. 191).
fn segment_tolerance(lower_h: f64) -> f64 {
    if lower_h < SPLIT_HEIGHTS[0] {
        0.001
    } else {
        0.01
    }
}

/// Adaptive Kronrod G7-K15 quadrature of `f` over `[a, b]`, m (ICD Annex F.2.6).
///
/// The K15 estimate is accepted once it agrees with the embedded G7 one to a relative `tolerance`,
/// or to an absolute `tolerance` in the ICD's units of m⁻³·km; otherwise the interval is halved,
/// down to [`MAX_RECURSION`] levels, where the K15 estimate is taken as it stands. The bound is
/// on depth, not work: an interval holding a jump in the density (the Eq. 109/110 thickness
/// switches) never meets a relative tolerance and is halved until rounding merges its nodes,
/// 41 levels deep on the Annex E fixture, as in the reference.
fn kronrod(f: &impl Fn(f64) -> f64, a: f64, b: f64, tolerance: f64, level: u32) -> f64 {
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut k15 = 0.0;
    let mut g7 = 0.0;
    for (i, (&x, &w)) in K15_NODES.iter().zip(&K15_WEIGHTS).enumerate() {
        let y = f(mid + half * x);
        k15 += y * w;
        if i % 2 == 1 {
            g7 += y * G7_WEIGHTS[i / 2];
        }
    }
    k15 *= half;
    g7 *= half;

    let accepted =
        libm::fabs((k15 - g7) / k15) <= tolerance || libm::fabs(k15 - g7) * 1.0e-3 <= tolerance;
    // A non-finite estimate does not improve by halving, and would recurse to the depth limit on
    // every branch: 2^50 intervals.
    if accepted || !k15.is_finite() || level == MAX_RECURSION {
        k15
    } else {
        kronrod(f, a, a + half, tolerance, level + 1)
            + kronrod(f, a + half, b, tolerance, level + 1)
    }
}

/// The Galileo specialisation of the electron-density profile hooks.
struct GalileoVariant;

impl Variant for GalileoVariant {
    const AMPLITUDE_RULES: AmplitudeRules = AmplitudeRules::Galileo;

    /// Whether the Chapman correction applies at height `h_in_km` (Galileo: below 100 km).
    fn apply_chapman_parameters(&self, h_in_km: f64) -> bool {
        h_in_km < 100.0
    }

    /// Galileo exponential arguments for the three Epstein layers (Eq. 111 to 113).
    fn compute_exponential_arguments(&self, h: f64, params: &NeQuickParameters) -> [f64; 3] {
        // Below 100 km the numerator height is clamped, but the B parameters still use h.
        let clipped_h = h.max(100.0);
        let exp = clip_exp(10.0 / (1.0 + libm::fabs(clipped_h - params.hm_f2)));
        [
            (clipped_h - params.hm_f2) / params.b2_bot,
            ((clipped_h - params.hm_f1) / params.bf1(h)) * exp,
            ((clipped_h - params.hm_e) / params.be(h)) * exp,
        ]
    }

    /// Galileo topside thickness parameter H0 (Eq. 99 to 106).
    fn compute_h0(&self, params: &NeQuickParameters, month: u32) -> f64 {
        // Auxiliary parameter ka (Eq. 99, 100).
        let ka = if month > 3 && month < 10 {
            6.705 - 0.014 * params.azr - 0.008 * params.hm_f2
        } else {
            let ratio = params.hm_f2 / params.b2_bot;
            -7.77 + 0.097 * ratio * ratio + 0.153 * params.nm_f2
        };

        // Auxiliary parameter kb (Eq. 101, 102).
        let kb = join(ka, 2.0, 1.0, ka - 2.0);
        let kb = join(8.0, kb, 1.0, kb - 8.0);

        // Auxiliary parameter Ha (Eq. 103) and x, v (Eq. 104, 105).
        let h_a = kb * params.b2_bot;
        let x = 0.01 * (h_a - 150.0);
        let v = (0.041163 * x - 0.183981) * x + 1.424472;

        // Topside thickness parameter (Eq. 106).
        h_a / v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (G2) Eq. 17/18 and the section 3.3 clamp: Az = 63.7 for the all-zero triple, else
    /// a0 + a1·μ + a2·μ² clamped to [0, 400].
    #[test]
    #[allow(clippy::float_cmp)] // the sentinel substitutes a literal, so exactness is the claim
    fn the_all_zero_alpha_sentinel_is_the_only_thing_that_yields_the_default_az() {
        // Eq. 17: an all-zero broadcast triple means "no ionospheric correction available", and the
        // reference substitutes Az = 63.7. Every STEC fixture broadcasts a real triple, so neither
        // the sentinel nor its conjunction was ever evaluated — and `||` in place of either `&&`
        // makes any single zero coefficient trigger the default.
        let sentinel = NeQuickGalileo::new([0.0, 0.0, 0.0]);
        assert_eq!(sentinel.effective_ionization_level(12.0), 63.7);
        assert_eq!(sentinel.effective_ionization_level(-45.0), 63.7);

        // A first coefficient that is not the default, with the other two zero: the polynomial is
        // constant at a0, and only a conjunction distinguishes this from the sentinel.
        let constant = NeQuickGalileo::new([100.0, 0.0, 0.0]);
        assert_eq!(constant.effective_ionization_level(12.0), 100.0);
        // A zero leading coefficient with a live slope is likewise not the sentinel.
        let sloped = NeQuickGalileo::new([0.0, 2.0, 0.0]);
        assert_eq!(sloped.effective_ionization_level(12.0), 24.0);
        assert_eq!(sloped.effective_ionization_level(-45.0), 0.0); // clamped from -90
        // Section 3.3's upper bound: 100 + 2·75 + 0.04·75² = 475 is clamped to 400, while
        // 100 + 2·50 + 0.04·50² = 300 is inside the operational range and stands.
        let quadratic = NeQuickGalileo::new([100.0, 2.0, 0.04]);
        assert_eq!(quadratic.effective_ionization_level(75.0), 400.0);
        assert!((quadratic.effective_ionization_level(50.0) - 300.0).abs() < 1.0e-12);
    }

    /// (G2) The Kronrod K15 nodes and weights and the embedded Gauss G7 weights are the values their
    /// definitions give: the G7 nodes are the roots of P7 and its weights `2/((1 − x²)·P7'(x)²)`,
    /// the eight further K15 nodes are the roots of the Stieltjes polynomial E8 orthogonal to
    /// `P7·xᵏ` for k = 0..7, and the K15 weights those of the interpolatory rule on all fifteen.
    ///
    /// The literals are the 40-digit values `derivations/kronrod_g7k15.mac` prints for x ≥ 0, where
    /// it also shows K15 exact through degree 22 and G7 through 13, and every one of the ICD's
    /// 33-digit Annex F.2.6 values within 5e-34 of the definition; `derivations/kronrod_g7k15.gp`
    /// recomputes all of it at 80 digits. Both literal sets round to the same doubles, so the
    /// stored tables are equal to the defined values bit for bit, mirrored about zero.
    #[test]
    #[expect(
        clippy::excessive_precision,
        reason = "the derivation's 40-digit values, kept whole"
    )]
    #[allow(clippy::float_cmp)] // equal to the nearest double is the claim
    fn the_kronrod_nodes_and_weights_are_their_defining_values() {
        const NODES: [f64; 8] = [
            0.0,
            0.207_784_955_007_898_467_600_689_403_773_244_913_479_8,
            0.405_845_151_377_397_166_906_606_412_076_961_463_347_4,
            0.586_087_235_467_691_130_294_144_838_258_729_598_436_8,
            0.741_531_185_599_394_439_863_864_773_280_788_407_074_1,
            0.864_864_423_359_769_072_789_712_788_640_926_201_211,
            0.949_107_912_342_758_524_526_189_684_047_851_262_400_8,
            0.991_455_371_120_812_639_206_854_697_526_328_516_642,
        ];
        const K15: [f64; 8] = [
            0.209_482_141_084_727_828_012_999_174_891_714_263_697_8,
            0.204_432_940_075_298_892_414_161_999_234_649_084_716_5,
            0.190_350_578_064_785_409_913_256_402_421_013_682_826_1,
            0.169_004_726_639_267_902_826_583_426_598_550_284_106_2,
            0.140_653_259_715_525_918_745_189_590_510_237_920_399_9,
            0.104_790_010_322_250_183_839_876_322_541_518_017_443_8,
            0.063_092_092_629_978_553_290_700_663_189_204_286_665_07,
            0.022_935_322_010_529_224_963_732_008_058_969_591_993_56,
        ];
        const G7: [f64; 4] = [
            0.417_959_183_673_469_387_755_102_040_816_326_530_612_2,
            0.381_830_050_505_118_944_950_369_775_488_975_133_878_4,
            0.279_705_391_489_276_667_901_467_771_423_779_582_486_9,
            0.129_484_966_168_869_693_270_611_432_679_082_018_328_6,
        ];
        for i in 0..8 {
            assert_eq!(K15_NODES[7 + i], NODES[i], "K15 node {}", 7 + i);
            assert_eq!(K15_NODES[7 - i], -NODES[i], "K15 node {}", 7 - i);
            assert_eq!(K15_WEIGHTS[7 + i], K15[i], "K15 weight {}", 7 + i);
            assert_eq!(K15_WEIGHTS[7 - i], K15[i], "K15 weight {}", 7 - i);
        }
        for i in 0..4 {
            assert_eq!(G7_WEIGHTS[3 + i], G7[i], "G7 weight {}", 3 + i);
            assert_eq!(G7_WEIGHTS[3 - i], G7[i], "G7 weight {}", 3 - i);
        }
        // The centre G7 weight is 512/1225 exactly (G3).
        assert_eq!(G7_WEIGHTS[3], 512.0 / 1225.0);
    }

    /// Eq. 99/100 select the summer form of `ka` for April to September, and March and October are
    /// the two ends nothing evaluates.
    ///
    /// `month > 3 && month < 10` admits {4..9}; the two mutants extend that to {3..9} and {4..10}.
    /// Every fixture in the crate asks for month 1, 4 or 11, so both boundaries are unreachable by
    /// construction -- a pure coverage hole in a seasonal switch whose two branches are far apart:
    /// Eq. 99 is linear in Azr and hmF2 while Eq. 100 is quadratic in hmF2/B2bot, and at the
    /// fixture-like profile below they differ by a factor of about 1.9 in the topside thickness.
    ///
    /// The assertion is a separation from the summer value rather than a literal, so it states
    /// which branch was taken and not the arithmetic inside it.
    #[test]
    #[allow(clippy::float_cmp)] // one branch taken twice returns the same number, bit for bit
    fn march_and_october_take_the_winter_form_of_the_topside_thickness() {
        let p = NeQuickParameters {
            azr: 50.0,
            nm_f2: 8.0,
            hm_f2: 300.0,
            hm_f1: 210.0,
            hm_e: 120.0,
            b2_bot: 45.0,
            b1_top: 27.0,
            b1_bot: 45.0,
            be_top: 45.0,
            be_bot: 5.0,
            amplitudes: [32.0, 1.0, 3.0],
        };
        let summer = GalileoVariant.compute_h0(&p, 6);
        for month in [3_u32, 10] {
            let h0 = GalileoVariant.compute_h0(&p, month);
            assert!(
                (h0 - summer).abs() > 1.0,
                "month {month} took the summer branch: {h0} against {summer}"
            );
        }
        // ...and the interior of the window really is one branch.
        for month in [4_u32, 6, 9] {
            assert_eq!(GalileoVariant.compute_h0(&p, month), summer);
        }
    }

    /// G7-K15 is exact for polynomials up to degree 13 in both rules, so a degree-13 integrand
    /// is accepted at the first level and returns the integral to rounding.
    #[test]
    fn kronrod_is_exact_on_a_degree_thirteen_polynomial() {
        let f = |x: f64| 3.0 * x.powi(13) - x.powi(6) + 2.0;
        let exact = |x: f64| 3.0 * x.powi(14) / 14.0 - x.powi(7) / 7.0 + 2.0 * x;
        let got = kronrod(&f, -0.5, 1.5, 1.0e-12, 0);
        let want = exact(1.5) - exact(-0.5);
        assert!((got - want).abs() < 1.0e-13 * want.abs(), "{got} vs {want}");
    }

    /// A peaked integrand fails the first G7/K15 comparison and must be bisected to reach the
    /// tolerance: an Epstein-like `sech²` layer of width 1 on a 1000-wide interval.
    #[test]
    fn kronrod_bisects_a_peaked_integrand_to_the_tolerance() {
        let f = |x: f64| {
            let c = libm::cosh(x - 300.0);
            1.0 / (c * c)
        };
        // ∫ sech²(x − 300) dx over [0, 1000] = tanh(700) + tanh(300) = 2 to binary64.
        let got = kronrod(&f, 0.0, 1000.0, 1.0e-6, 0);
        assert!((got - 2.0).abs() < 1.0e-5, "{got}");
        // The single-level K15 estimate is far off, so the bisection is what got it there.
        let one_level = kronrod(&f, 0.0, 1000.0, 1.0e-6, MAX_RECURSION);
        assert!((one_level - 2.0).abs() > 0.1, "{one_level}");
    }

    /// At the depth limit the K15 estimate is taken as it stands, however far it is from the G7 one.
    ///
    /// A negative tolerance is never met, so starting three levels short of the limit must halve
    /// exactly three times and return the sum of the eight K15 estimates at the limit.
    #[test]
    fn kronrod_stops_halving_at_the_depth_limit() {
        let f = |x: f64| libm::exp(x);
        let got = kronrod(&f, 0.0, 8.0, -1.0, MAX_RECURSION - 3);
        let leaves: f64 = (0..8)
            .map(|i| kronrod(&f, f64::from(i), f64::from(i + 1), -1.0, MAX_RECURSION))
            .sum();
        assert!((got - leaves).abs() < 1.0e-12 * leaves, "{got} vs {leaves}");
        // The leaves really are the limit's single-interval estimates, not converged integrals.
        let one = kronrod(&f, 0.0, 8.0, -1.0, MAX_RECURSION);
        assert!((one - got).abs() > 0.0);
    }

    /// A NaN coordinate yields a NaN STEC, at once, rather than 2^50 halvings.
    #[test]
    fn a_nan_input_ends_the_quadrature_at_once() {
        let f = |_: f64| f64::NAN;
        assert!(kronrod(&f, 0.0, 1.0, 1.0e-3, 0).is_nan());
        let stec = NeQuickGalileo::new([236.831641, -0.39362878, 0.00402826613]).stec(
            DateTimeComponents::new(4, 0.0),
            GeodeticPoint::new(f64::NAN, 0.2, 0.0),
            GeodeticPoint::new(0.8, 0.3, 2.0e7),
        );
        assert!(stec.is_nan(), "{stec}");
    }

    /// Eq. 188 to 190 split at 1000 and 2000 km only where the path crosses them; an end point
    /// on a split height does not split.
    #[test]
    fn the_path_splits_only_at_heights_it_strictly_crosses() {
        let got = |rec: f64, sat: f64| splits(rec, sat).collect::<Vec<_>>();
        assert_eq!(got(0.0, 2.0e7), [1.0e6, 2.0e6]);
        assert_eq!(got(-30.0, 1.5e6), [1.0e6]);
        assert_eq!(got(1.0e6, 2.0e7), [2.0e6]);
        assert_eq!(got(0.0, 1.0e6), [] as [f64; 0]);
        assert_eq!(got(1.2e6, 2.0e6), [] as [f64; 0]);
        assert_eq!(got(2.0e6, 2.0e7), [] as [f64; 0]);
    }

    /// Eq. 191: a segment starting below 1000 km integrates to 1e-3, one starting at or above it
    /// to 1e-2.
    #[test]
    #[allow(clippy::float_cmp)] // the tolerances are literals, selected not computed
    fn segments_from_one_thousand_kilometres_up_take_the_loose_tolerance() {
        assert_eq!(segment_tolerance(0.0), 0.001);
        assert_eq!(segment_tolerance(999_999.0), 0.001);
        assert_eq!(segment_tolerance(1_000_000.0), 0.01);
        assert_eq!(segment_tolerance(2_000_000.0), 0.01);
    }
}
