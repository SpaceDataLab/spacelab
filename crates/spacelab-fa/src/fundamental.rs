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
//! published algorithm specifies; the planetary longitudes modulo `2π`. Every
//! function is cross-checked against `pyerfa` (ERFA, BSD-3-Clause) in the tests
//! below and in `derivations/fundamental_args.py`, to under `1e-13` rad.

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
    #![allow(clippy::excessive_precision)] // verbatim pyerfa oracle values
    #![allow(clippy::type_complexity)] // a test table of (name, fn ptr, expected)

    use super::*;

    /// Reference values from `pyerfa` (ERFA, BSD) at `t = 0.06` Julian centuries
    /// (TT JD 2400000.5 + 53736.0), the SOFA self-test epoch. A truncated
    /// coefficient shifts the argument by more than this tolerance and fails here.
    /// See `derivations/fundamental_args.py`.
    #[test]
    fn matches_pyerfa_at_the_sofa_epoch() {
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
        for (name, f, expected) in cases {
            let got = f(t);
            assert!(
                (got - expected).abs() < 1e-13,
                "{name}: got {got:.17e}, expected {expected:.17e}, delta {:.2e}",
                got - expected
            );
        }
    }

    /// The same fourteen against `pyerfa`, away from the self-test epoch.
    ///
    /// One small `t` cannot separate a polynomial's coefficients: at `t = 0.06` the quartic term is
    /// five orders below the linear one, so `faom03`'s `-0.00005939` moves the argument by 7.5e-15
    /// rad -- under the tolerance above, and under a `f64` ulp of the argument itself. Over a
    /// century either side of J2000, and two centuries out, the high-order terms carry their own
    /// weight: the same sign change is 5.8e-10 rad at `t = 1` and 9.2e-9 at `t = 2`.
    #[test]
    #[allow(clippy::too_many_lines)] // seventy oracle values: five epochs of fourteen arguments
    fn matches_pyerfa_across_four_centuries() {
        let names = [
            "fal03", "falp03", "faf03", "fad03", "faom03", "fame03", "fave03", "fae03", "fama03",
            "faju03", "fasa03", "faur03", "fane03", "fapa03",
        ];
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
        // Both evaluate the same Horner form on the same constants and reduce with the same `fmod`,
        // so the agreement is exact at all seventy points; the two ulps are there for a future
        // toolchain that contracts a multiply-add, not for an error anything here can make.
        for (t, expected) in cases {
            for i in 0..14 {
                spacelab_testkit::assert_within_ulps(
                    fns[i](t),
                    expected[i],
                    2,
                    &format!("{} at t = {t}", names[i]),
                );
            }
        }
    }
}
