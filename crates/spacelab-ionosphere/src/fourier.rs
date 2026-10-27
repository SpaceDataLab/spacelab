//! Fourier-in-universal-time evaluation of the CCIR coefficients.
//!
//! Ported from Orekit's `FourierTimeSeries`. Given the effective ionisation level
//! `az` and the universal time, it collapses each monthly CCIR coefficient block into
//! the `cf2` (76) and `cm3` (49) spatial-coefficient arrays consumed by the profile
//! parameters, after interpolating between the low/high solar-activity rows using the
//! effective sunspot number `azr`.
//!
//! The collapse is `Σ_d ((1 − azr01)·loᵈ + azr01·hiᵈ)·T_d(h)`, which is **affine in `azr01`**, the
//! only factor that depends on `az`. Splitting it at that seam — [`EpochSeries`] holds the two harmonic sums,
//! which depend only on the epoch — turns the per-evaluation cost from `2·depth` multiply-adds per
//! column into one. The Galileo path needs that split: its Az is a function of the local MODIP, so
//! it re-evaluates the series at every Gauss node of every rung of the STEC ladder, where the ITU
//! path evaluates it once per ray.
//!
//! Gate: `derivations/nequick_fourier.mac` reads the CCIR files as exact rationals and shows that
//! Eq. 49's time angle is the form the code evaluates, that the code's flat harmonic walk is
//! Eq. 50/51's index pattern, that blending the solar-activity rows before or after the series is
//! the same (the `EpochSeries` split, all 125 columns of April and November) and that Eq. 19 is
//! the root of its defining quadratic; `derivations/nequick_fourier.gp` re-reads the files and
//! recomputes its 30-digit collapse values. The module test
//! `the_collapse_is_the_icd_fourier_series_of_the_ccir_files` (G2) holds the code to them.

use std::f64::consts::PI;

use crate::ccir::CcirCoefficients;

/// F2 spatial coefficients per epoch.
const F2_COLUMNS: usize = 76;

/// M(3000)F2 spatial coefficients per epoch.
const FM3_COLUMNS: usize = 49;

/// The epoch-only half of the CCIR collapse: each block's low- and high-solar-activity rows,
/// already summed over their time harmonics and awaiting only the `azr` interpolation.
pub(crate) struct EpochSeries {
    /// foF2 harmonic sums, low then high solar activity (76 values each).
    f2: [Vec<f64>; 2],
    /// M(3000)F2 harmonic sums, low then high solar activity (49 values each).
    m3: [Vec<f64>; 2],
}

impl EpochSeries {
    /// Sum both solar-activity rows of each block over the time harmonics at `hours`.
    pub(crate) fn new(hours: f64, coeffs: &CcirCoefficients) -> Self {
        // Time argument (Eq. 49) and its sine/cosine harmonics up to order 6.
        let t = (15.0 * hours).to_radians() - PI;
        let sc_t = sin_cos(t, 6);

        Self {
            f2: harmonic_sums(&coeffs.flatten_f2, &sc_t, F2_COLUMNS, 13),
            m3: harmonic_sums(&coeffs.flatten_fm3, &sc_t, FM3_COLUMNS, 9),
        }
    }
}

/// Fourier time series evaluated at one epoch and effective ionisation level.
pub(crate) struct FourierTimeSeries {
    /// Effective ionisation level (Az).
    pub az: f64,
    /// Effective sunspot number (Azr, Eq. 19).
    pub azr: f64,
    /// Spatial coefficients for foF2 (76 values).
    pub cf2: Vec<f64>,
    /// Spatial coefficients for M(3000)F2 (49 values).
    pub cm3: Vec<f64>,
}

impl FourierTimeSeries {
    /// Evaluate at `az` against a series whose epoch half is already summed.
    pub(crate) fn at(az: f64, epoch: &EpochSeries) -> Self {
        let mut series = Self {
            az,
            azr: 0.0,
            cf2: vec![0.0; F2_COLUMNS],
            cm3: vec![0.0; FM3_COLUMNS],
        };
        series.set_az(az, epoch);
        series
    }

    /// Re-interpolate in place for a new `az`, reusing the coefficient buffers.
    ///
    /// The Galileo STEC ladder calls this once per Gauss node and can reach 1.7e7 of them, so the
    /// two allocations [`Self::at`] makes are themselves worth hoisting out of the loop.
    pub(crate) fn set_az(&mut self, az: f64, epoch: &EpochSeries) {
        // Effective sunspot number (Eq. 19).
        let azr = libm::sqrt(167_273.0 + (az - 63.7) * 1123.6) - 408.99;
        let azr01 = azr * 0.01;
        let omazr01 = 1.0 - azr01;

        self.az = az;
        self.azr = azr;
        interpolate(&mut self.cf2, &epoch.f2, omazr01, azr01);
        interpolate(&mut self.cm3, &epoch.m3, omazr01, azr01);
    }
}

/// `out[c] = omazr01·rows[0][c] + azr01·rows[1][c]` — Eq. 44 merged with Eq. 50/51.
fn interpolate(out: &mut [f64], rows: &[Vec<f64>; 2], omazr01: f64, azr01: f64) {
    for (c, value) in out.iter_mut().enumerate() {
        *value = omazr01 * rows[0][c] + azr01 * rows[1][c];
    }
}

/// Sum a flattened `[column][harmonic][row]` block over its harmonics, per solar-activity row.
///
/// For each column the constant term plus `depth - 1` sine/cosine harmonics are summed. The two
/// rows are kept apart rather than blended here, because the blend is the only part that depends
/// on the effective ionisation level.
fn harmonic_sums(flatten: &[f64], sc_t: &[f64], columns: usize, depth: usize) -> [Vec<f64>; 2] {
    let stride = 2 * depth;
    let mut rows = [vec![0.0; columns], vec![0.0; columns]];
    for (row, out) in rows.iter_mut().enumerate() {
        for (column, value) in out.iter_mut().enumerate() {
            let base = column * stride + row;
            // Harmonic 0 is the constant term.
            let mut acc = flatten[base];
            for d in 1..depth {
                acc += flatten[base + 2 * d] * sc_t[d - 1];
            }
            *value = acc;
        }
    }
    rows
}

/// Return `[sin(a), cos(a), sin(2a), cos(2a), …, sin(na), cos(na)]`.
pub(crate) fn sin_cos(a: f64, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(2 * n);
    for k in 1..=n {
        let (s, c) = libm::sincos(k as f64 * a);
        out.push(s);
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccir;

    /// (G2) The collapse is ICD Eq. 19, 44, 46 and 49 to 51 evaluated on the shipped CCIR files.
    ///
    /// The references are `derivations/nequick_fourier.mac`'s exact evaluation of those equations
    /// as printed, on each file's decimals read as exact rationals in the reference's Fortran order
    /// (harmonic fastest, then column, then solar-activity row), at 30 digits; there it also shows
    /// that the code's split -- harmonic sums per row, blended after -- is the ICD's blend-then-sum
    /// exactly, and `derivations/nequick_fourier.gp` recomputes the values at 60 digits. The epochs
    /// are April at 14 UT (T = π/6) and November at 03:30 UT (T = −17π/24), with Az = 120 and 200;
    /// the columns span the constant, sine and cosine slots of the first, a middle and the last
    /// coefficient, so a permuted layout or a shifted harmonic shows.
    #[test]
    #[expect(
        clippy::excessive_precision,
        reason = "the derivation's 30-digit values, kept whole"
    )]
    fn the_collapse_is_the_icd_fourier_series_of_the_ccir_files() {
        const F2_COLUMNS: [usize; 8] = [1, 2, 12, 13, 14, 40, 75, 76];
        const M3_COLUMNS: [usize; 7] = [1, 2, 7, 8, 9, 30, 49];
        // (month, UT hours, Az, Azr, cf2 at F2_COLUMNS, cm3 at M3_COLUMNS).
        type Case = (u32, f64, f64, f64, [f64; 8], [f64; 7]);
        let cases: [Case; 2] = [
            (
                4,
                14.0,
                120.0,
                71.147_147_073_625_204_211_266_046_141_7,
                [
                    9.122_235_936_040_195_949_881_466_895_93,
                    0.204_212_382_122_895_880_385_204_145_946,
                    -99.082_312_589_234_103_250_387_583_700_8,
                    2.055_445_944_531_572_258_577_301_500_81,
                    0.851_630_079_410_417_149_926_445_335_472,
                    2.343_773_174_960_281_900_148_964_133_34,
                    -0.014_781_480_904_526_359_623_634_447_616,
                    -0.027_257_486_579_873_965_550_260_710_163_7,
                ],
                [
                    2.798_828_056_891_021_549_934_778_084_98,
                    -0.004_509_437_012_510_932_319_182_035_936_33,
                    0.688_014_772_571_126_752_667_163_770_067,
                    -0.372_045_977_159_391_120_901_205_773_178,
                    -0.099_144_954_870_895_171_871_297_877_145_6,
                    -1.558_018_747_546_697_341_208_709_043_93,
                    -0.015_182_599_628_742_794_513_009_322_678_9,
                ],
            ),
            (
                11,
                3.5,
                200.0,
                157.066_251_621_691_746_509_393_615_302,
                [
                    10.972_804_560_523_860_294_133_176_843_5,
                    1.662_780_532_126_754_564_998_858_482_35,
                    60.731_006_078_737_520_470_474_590_356_4,
                    -2.123_851_463_095_329_446_282_966_225_18,
                    1.164_499_806_941_378_880_422_774_928_73,
                    1.098_811_168_920_666_458_096_034_157_7,
                    -0.133_278_786_554_370_562_432_571_720_446,
                    0.070_081_534_283_870_252_091_842_599_329_7,
                ],
                [
                    2.472_388_644_747_346_540_050_278_871_28,
                    0.197_104_744_476_384_855_600_274_894_708,
                    0.336_970_444_466_532_078_785_576_835_126,
                    0.469_697_848_917_377_625_591_586_484_224,
                    0.038_216_223_367_268_051_314_440_559_026_1,
                    0.248_554_406_595_724_591_554_708_649_932,
                    0.024_403_056_461_118_298_954_959_353_601_5,
                ],
            ),
        ];
        let close = |got: f64, want: f64| (got - want).abs() <= 1.0e-13 * want.abs().max(1.0);
        for (month, hours, az, azr, cf2, cm3) in cases {
            let series = FourierTimeSeries::at(az, &EpochSeries::new(hours, ccir::load(month)));
            assert!(
                close(series.azr, azr),
                "month {month}: Azr {} vs {azr}",
                series.azr
            );
            for (&c, &want) in F2_COLUMNS.iter().zip(&cf2) {
                let got = series.cf2[c - 1];
                assert!(close(got, want), "month {month}, cf2 {c}: {got} vs {want}");
            }
            for (&c, &want) in M3_COLUMNS.iter().zip(&cm3) {
                let got = series.cm3[c - 1];
                assert!(close(got, want), "month {month}, cm3 {c}: {got} vs {want}");
            }
        }
    }
}
