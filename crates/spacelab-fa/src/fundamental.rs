//! Fundamental arguments of the IAU 2006/2000A precession-nutation theory.
//!
//! The five luni-solar Delaunay arguments (Simon et al. 1994, adopted by the IERS
//! Conventions 2003/2010), the eight planetary mean longitudes and the general
//! accumulated precession in longitude `pA` (Souchay et al. 1999 / Kinoshita &
//! Souchay). Each is a polynomial in `t`, Julian centuries of TDB since J2000.0
//! (TT is used in practice — the TDB−TT difference is below 2 µs, well under a
//! microarcsecond in these arguments), returning **radians**.
//!
//! These are the phase arguments summed with integer multipliers to form the CIP
//! `X, Y` and the CIO locator `s` series ([`crate::cio`]).
//!
//! # Provenance
//!
//! The polynomial coefficients are the IERS-published decimals of the cited models
//! — not transcribed from any implementation's source. The Delaunay arguments are
//! reduced modulo a full turn in arcseconds before conversion, exactly as the
//! published algorithm specifies; the planetary longitudes modulo `2π`.
//! `derivations/fundamental_arguments.mac` holds every coefficient to the published
//! decimal and writes the exact values the tests hold each function to; `pyerfa`
//! (ERFA, BSD-3-Clause) is a comparator.

use spacelab_core::Angle;

/// Arcseconds to radians, `π / 648000` (exact to the last bit — no truncated literal).
pub(crate) const DAS2R: f64 = core::f64::consts::PI / 648_000.0;
/// Arcseconds in one full turn, `360 × 3600`.
const TURNAS: f64 = 1_296_000.0;

/// Mean anomaly of the Moon, `l` (radians).
///
/// Delaunay argument; IERS Conventions (2003), from Simon et al. (1994).
#[must_use]
pub fn fal03(t: f64) -> f64 {
    let a = 485_868.249_036
        + t * (1_717_915_923.217_8 + t * (31.879_2 + t * (0.051_635 + t * -0.000_244_70)));
    libm::fmod(a, TURNAS) * DAS2R
}

/// Mean anomaly of the Sun, `l'` (radians).
///
/// Delaunay argument; IERS Conventions (2003).
#[must_use]
pub fn falp03(t: f64) -> f64 {
    let a = 1_287_104.793_048
        + t * (129_596_581.048_1 + t * (-0.553_2 + t * (0.000_136 + t * -0.000_011_49)));
    libm::fmod(a, TURNAS) * DAS2R
}

/// Mean longitude of the Moon minus that of the ascending node, `F` (radians).
///
/// Delaunay argument; IERS Conventions (2003).
#[must_use]
pub fn faf03(t: f64) -> f64 {
    let a = 335_779.526_232
        + t * (1_739_527_262.847_8 + t * (-12.751_2 + t * (-0.001_037 + t * 0.000_004_17)));
    libm::fmod(a, TURNAS) * DAS2R
}

/// Mean elongation of the Moon from the Sun, `D` (radians).
///
/// Delaunay argument; IERS Conventions (2003).
#[must_use]
pub fn fad03(t: f64) -> f64 {
    let a = 1_072_260.703_692
        + t * (1_602_961_601.209_0 + t * (-6.370_6 + t * (0.006_593 + t * -0.000_031_69)));
    libm::fmod(a, TURNAS) * DAS2R
}

/// Mean longitude of the ascending node of the Moon, `Ω` (radians).
///
/// Delaunay argument; IERS Conventions (2003).
#[must_use]
pub fn faom03(t: f64) -> f64 {
    let a = 450_160.398_036
        + t * (-6_962_890.543_1 + t * (7.472_2 + t * (0.007_702 + t * -0.000_059_39)));
    libm::fmod(a, TURNAS) * DAS2R
}

/// Mean longitude of Mercury (radians). Souchay et al. (1999).
#[must_use]
pub fn fame03(t: f64) -> f64 {
    libm::fmod(4.402_608_842 + 2_608.790_314_157_4 * t, Angle::TAU)
}

/// Mean longitude of Venus (radians). Souchay et al. (1999).
#[must_use]
pub fn fave03(t: f64) -> f64 {
    libm::fmod(3.176_146_697 + 1_021.328_554_621_1 * t, Angle::TAU)
}

/// Mean longitude of Earth (radians). Souchay et al. (1999).
#[must_use]
pub fn fae03(t: f64) -> f64 {
    libm::fmod(1.753_470_314 + 628.307_584_999_1 * t, Angle::TAU)
}

/// Mean longitude of Mars (radians). Souchay et al. (1999).
#[must_use]
pub fn fama03(t: f64) -> f64 {
    libm::fmod(6.203_480_913 + 334.061_242_670_0 * t, Angle::TAU)
}

/// Mean longitude of Jupiter (radians). Souchay et al. (1999).
#[must_use]
pub fn faju03(t: f64) -> f64 {
    libm::fmod(0.599_546_497 + 52.969_096_264_1 * t, Angle::TAU)
}

/// Mean longitude of Saturn (radians). Souchay et al. (1999).
#[must_use]
pub fn fasa03(t: f64) -> f64 {
    libm::fmod(0.874_016_757 + 21.329_910_496_0 * t, Angle::TAU)
}

/// Mean longitude of Uranus (radians). Souchay et al. (1999).
#[must_use]
pub fn faur03(t: f64) -> f64 {
    libm::fmod(5.481_293_872 + 7.478_159_856_7 * t, Angle::TAU)
}

/// Mean longitude of Neptune (radians). Kinoshita & Souchay (1990).
#[must_use]
pub fn fane03(t: f64) -> f64 {
    libm::fmod(5.311_886_287 + 3.813_303_563_8 * t, Angle::TAU)
}

/// General accumulated precession in longitude, `pA` (radians).
///
/// Kinoshita & Souchay; IERS Conventions (2003). No range reduction — the argument
/// stays small over any epoch of interest.
#[must_use]
pub fn fapa03(t: f64) -> f64 {
    (0.024_381_750 + 0.000_005_386_91 * t) * t
}

#[cfg(test)]
mod tests {
    // Exact values to 25 digits as the derivation prints them, and verbatim pyerfa values.
    #![allow(clippy::excessive_precision, clippy::unreadable_literal)]
    #![allow(clippy::type_complexity)] // a test table of (name, fn ptr, expected)

    use super::*;

    /// The fourteen arguments in the order `ARGUMENTS_EXACT` holds them.
    const ARGUMENTS: [fn(f64) -> f64; 14] = [
        fal03, falp03, faf03, fad03, faom03, fame03, fave03, fae03, fama03, faju03, fasa03, faur03,
        fane03, fapa03,
    ];

    /// `t = 0.06` (TT JD 2400000.5 + 53736.0, the SOFA self-test epoch), a century and half a
    /// century either side of J2000, and two centuries out.
    const EPOCHS: [f64; 6] = [0.06, -1.0, -0.5, 0.5, 1.0, 2.0];

    /// The fourteen at [`EPOCHS`]: eqs (5.43) and (5.44) of the IERS Conventions (2010) in exact
    /// arithmetic, reduced as the module says, to 25 digits (`derivations/fundamental_arguments.mac`
    /// reads this table back).
    const ARGUMENTS_EXACT: [[f64; 14]; 6] = [
        [
            5.705402575902754796387522e0,
            6.239065584464557534958239e0,
            4.981049699601457816763943e0,
            2.421971746329525499703946e-1,
            1.570165698203728253308263e-1,
            3.850395011954332284189684e0,
            1.624006902470132962942178e0,
            1.752813570868479743325185e0,
            1.114414244481653350533886e0,
            3.777692272845999882384979e0,
            2.153811386759999952638085e0,
            5.929983463401999983395149e0,
            5.540684500827999991532765e0,
            1.462924392875999945860204e-3,
        ],
        [
            -1.115183659436996646064090e0,
            -2.655231640888373072429724e-2,
            -6.086816721524464884260624e0,
            -1.616702487420072564323435e-1,
            4.523594802968210737714460e0,
            -3.148988143051198552931279e0,
            -2.763881610069907381035438e-1,
            -4.518769274320938784396610e0,
            -1.132125783661503199885088e0,
            -2.104067309663308184597706e0,
            -1.606337817461240569224140e0,
            -1.996865984700000000000000e0,
            1.498582723200000000000000e0,
            -2.437636309000000000000000e-2,
        ],
        [
            -2.521445155848006577830886e0,
            -3.483807756864621399410730e-2,
            -5.371033020586903672689138e0,
            -3.764779403781769519892617e0,
            2.114153037687785380819460e-1,
            -5.656374957705185753390926e0,
            -4.833306039183081845977059e0,
            -4.524242133750262630660948e0,
            -3.747507742510338076867831e0,
            -7.522604063316540922988529e-1,
            -3.507753183820413523074713e0,
            1.742213943650000000000000e0,
            3.405234505100000000000000e0,
            -1.218952827250000000000000e-2,
        ],
        [
            9.494486128694003103544561e-1,
            6.231771683214022492574395e0,
            2.343626966703829947999282e0,
            1.595326523953606538415562e0,
            -2.129704104629074497490256e0,
            1.895222027346012799540353e0,
            4.902414126003495369051772e0,
            1.747997454570676153735662e0,
            3.588098954151165123017257e0,
            1.951353400331654092298853e0,
            5.255786697820413523074713e0,
            2.937188493170413523074713e0,
            9.353527617204135230747132e-1,
            1.219222172750000000000000e-2,
        ],
        [
            5.826604253498133491669960e0,
            6.223481898965887114950875e0,
            3.059317938336139070532098e0,
            4.275356347494807376586291e0,
            -1.586439578169707548452508e-1,
            5.671020519871612076005992e0,
            3.454962478274042611782571e-1,
            1.742524595141352307471323e0,
            9.727169953023302460345146e-1,
            3.303160303663308184597706e0,
            3.354371331461240569224140e0,
            3.930831143408270461494265e-1,
            2.842004543620413523074713e0,
            2.438713691000000000000000e-2,
        ],
        [
            3.014778051161376177535047e0,
            6.206898310216130863927996e0,
            4.490607126128474610903657e0,
            3.352184524880349302489012e0,
            -2.499654439742786264705095e0,
            6.562468905636376750866970e-1,
            3.798031105834394999281801e0,
            1.731578876282704614942647e0,
            2.025138384784246968994316e0,
            6.006774110326616369195412e0,
            5.834725905922481138448279e0,
            1.588057663861240569224140e0,
            3.721228002408270461494265e-1,
            4.878504764000000000000000e-2,
        ],
    ];

    /// The worst miss measured is 1.593e-12 rad, `fal03` at `t = 2`: 0.69 of the 2.3e-12 rad ulp of
    /// its unreduced `3.5e9″` polynomial. The bound is 3.1 times that.
    const MEASURED_BOUND: f64 = 5.0e-12;

    /// Every argument is its published polynomial, reduced as specified, at six epochs. (G3)
    ///
    /// One small `t` cannot separate a polynomial's coefficients: at `t = 0.06` the quartic term is
    /// five orders below the linear one, so `faom03`'s `-0.00005939` moves the argument by 7.5e-15
    /// rad, under a `f64` ulp of the argument itself. Over a century either side of J2000, and two
    /// centuries out, the high-order terms carry their own weight: the same sign change is 5.8e-10
    /// rad at `t = 1` and 9.2e-9 at `t = 2`. The bound is the `f64` evaluation's own resolution
    /// there: two centuries out the Delaunay polynomials reach `3.5e9″`, whose ulp is `2.3e-12` rad.
    #[test]
    fn the_arguments_are_their_published_polynomials_reduced_as_specified() {
        let mut worst = (0.0_f64, String::new());
        for (t, exact) in EPOCHS.iter().zip(ARGUMENTS_EXACT.iter()) {
            for (i, (f, want)) in ARGUMENTS.iter().zip(exact.iter()).enumerate() {
                let miss = (f(*t) - want).abs();
                if miss > worst.0 || miss.is_nan() {
                    worst = (miss, format!("argument {i} at t = {t}"));
                }
            }
        }
        assert!(worst.0 < MEASURED_BOUND, "{:.3e} at {}", worst.0, worst.1);
    }

    /// The distances to `pyerfa` (ERFA, BSD) at `t = 0.06` Julian centuries (TT JD 2400000.5 +
    /// 53736.0), the SOFA self-test epoch.
    #[test]
    #[ignore = "comparator: the distance to pyerfa, not a verdict"]
    fn sofa_epoch_distance_to_pyerfa() {
        let t = 0.06;
        let cases: [(&str, fn(f64) -> f64, f64); 14] = [
            ("fal03", fal03, 5.705_402_575_902_765_78e0),
            ("falp03", falp03, 6.239_065_584_464_555_95e0),
            ("faf03", faf03, 4.981_049_699_601_498_13e0),
            ("fad03", fad03, 2.421_971_746_330_537_91e-1),
            ("faom03", faom03, 1.570_165_698_203_729_86e-1),
            ("fame03", fame03, 3.850_395_011_954_368_84e0),
            ("fave03", fave03, 1.624_006_902_470_128_47e0),
            ("fae03", fae03, 1.752_813_570_868_475_78e0),
            ("fama03", fama03, 1.114_414_244_481_654_49e0),
            ("faju03", faju03, 3.777_692_272_846_000_07e0),
            ("fasa03", fasa03, 2.153_811_386_760_000_17e0),
            ("faur03", faur03, 5.929_983_463_402_000_33e0),
            ("fane03", fane03, 5.540_684_500_828_000_13e0),
            ("fapa03", fapa03, 1.462_924_392_875_999_96e-3),
        ];
        let distances: Vec<f64> = cases
            .iter()
            .map(|(_, f, want)| (f(t) - want).abs())
            .collect();
        pyerfa_row(
            "the fourteen fundamental arguments at the SOFA epoch",
            &distances,
        );
    }

    /// The same fourteen against `pyerfa` at five epochs away from the self-test epoch.
    #[test]
    #[ignore = "comparator: the distance to pyerfa, not a verdict"]
    #[allow(clippy::too_many_lines)] // seventy oracle values: five epochs of fourteen arguments
    fn four_centuries_distance_to_pyerfa() {
        let fns: [fn(f64) -> f64; 14] = [
            fal03, falp03, faf03, fad03, faom03, fame03, fave03, fae03, fama03, faju03, fasa03,
            faur03, fane03, fapa03,
        ];
        let cases: [(f64, [f64; 14]); 5] = [
            (
                -1.0,
                [
                    -1.115_183_659_435_983_19e0,
                    -2.655_231_640_885_251_72e-2,
                    -6.086_816_721_524_314_74e0,
                    -1.616_702_487_427_102_71e-1,
                    4.523_594_802_968_212_92e0,
                    -3.148_988_143_051_425_94e0,
                    -2.763_881_610_070_058_06e-1,
                    -4.518_769_274_320_952_43e0,
                    -1.132_125_783_661_507_51e0,
                    -2.104_067_309_663_314_23e0,
                    -1.606_337_817_461_241_71e0,
                    -1.996_865_984_699_999_43e0,
                    1.498_582_723_200_000_19e0,
                    -2.437_636_309_000_000_21e-2,
                ],
            ),
            (
                -0.5,
                [
                    -2.521_445_155_847_714_23e0,
                    -3.483_807_756_862_915_27e-2,
                    -5.371_033_020_586_909_68e0,
                    -3.764_779_403_781_999_27e0,
                    2.114_153_037_687_796_50e-1,
                    -5.656_374_957_705_267_55e0,
                    -4.833_306_039_183_128_18e0,
                    -4.524_242_133_750_227_88e0,
                    -3.747_507_742_510_357_79e0,
                    -7.522_604_063_316_578_05e-1,
                    -3.507_753_183_820_414_12e0,
                    1.742_213_943_650_000_37e0,
                    3.405_234_505_100_000_14e0,
                    -1.218_952_827_249_999_98e-2,
                ],
            ),
            (
                0.5,
                [
                    9.494_486_128_689_610_95e-1,
                    6.231_771_683_214_034_85e0,
                    2.343_626_966_703_756_59e0,
                    1.595_326_523_953_610_96e0,
                    -2.129_704_104_629_076_78e0,
                    1.895_222_027_346_221_69e0,
                    4.902_414_126_003_556_31e0,
                    1.747_997_454_570_693_78e0,
                    3.588_098_954_151_170_97e0,
                    1.951_353_400_331_655_05e0,
                    5.255_786_697_820_413_52e0,
                    2.937_188_493_170_413_75e0,
                    9.353_527_617_204_138_17e-1,
                    1.219_222_172_750_000_08e-2,
                ],
            ),
            (
                1.0,
                [
                    5.826_604_253_498_457_45e0,
                    6.223_481_898_965_822_27e0,
                    3.059_317_938_336_432_62e0,
                    4.275_356_347_495_069_54e0,
                    -1.586_439_578_169_722_70e-1,
                    5.671_020_519_871_966_31e0,
                    3.454_962_478_273_770_90e-1,
                    1.742_524_595_141_304_64e0,
                    9.727_169_953_023_775_41e-1,
                    3.303_160_303_663_311_48e0,
                    3.354_371_331_461_241_11e0,
                    3.930_831_143_408_273_22e-1,
                    2.842_004_543_620_413_77e0,
                    2.438_713_690_999_999_92e-2,
                ],
            ),
            (
                2.0,
                [
                    3.014_778_051_159_783_70e0,
                    6.206_898_310_215_986_92e0,
                    4.490_607_126_128_377_76e0,
                    3.352_184_524_881_871_57e0,
                    -2.499_654_439_742_796_89e0,
                    6.562_468_905_638_283_45e-1,
                    3.798_031_105_834_418_50e0,
                    1.731_578_876_282_640_07e0,
                    2.025_138_384_784_320_07e0,
                    6.006_774_110_326_631_44e0,
                    5.834_725_905_922_482_53e0,
                    1.588_057_663_861_242_47e0,
                    3.721_228_002_408_274_45e-1,
                    4.878_504_764_000_000_24e-2,
                ],
            ),
        ];
        let distances: Vec<f64> = cases
            .iter()
            .flat_map(|(t, expected)| {
                fns.iter()
                    .zip(expected)
                    .map(|(f, want)| (f(*t) - want).abs())
            })
            .collect();
        pyerfa_row(
            "the fourteen fundamental arguments at t = -1, -0.5, 0.5, 1, 2",
            &distances,
        );
    }

    /// One `comparison | ...` row of `tools/compare.py`, in radians.
    fn pyerfa_row(quantity: &str, distances: &[f64]) {
        let max = distances.iter().copied().fold(0.0, f64::max);
        #[expect(
            clippy::cast_precision_loss,
            reason = "a count of at most seventy distances"
        )]
        let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
        println!(
            "comparison | pyerfa | not recorded | {quantity} | rad | {} | {max:.3e} | {rms:.3e}",
            distances.len()
        );
    }
}
