//! IERS 2010 sub-daily (diurnal and semidiurnal) EOP corrections.
//!
//! The daily `finals2000A` series is a smoothed record; the real pole and rotation carry
//! diurnal and semidiurnal signals the daily samples cannot represent. The IERS Conventions
//! (2010) prescribe adding three high-frequency models on top of the interpolated series:
//!
//! * **ocean tides** in `x_p`, `y_p`, UT1 — the Ray model (`ortho_eop`, Ch. 8.2),
//! * **lunisolar libration** in polar motion — Table 5.1a (`pmsdnut2_terms`),
//! * **lunisolar libration** in UT1 and LOD — Table 5.1b (`utlibr_terms`).
//!
//! Magnitudes: hundreds of µas in polar motion, tens of µs in UT1 — decimetres at the surface.
//! Negligible against a TLE's own kilometre-scale error, but the IERS reduction is not complete
//! without them, so a caller that has better-than-mas inputs can apply [`subdaily_correction`].
//!
//! # Provenance
//!
//! Derived work based on the IERS Conventions software (`ORTHO_EOP.F`, `CNMTX.F`, `PMSDNUT2.F`,
//! `UTLIBR.F`, `FUNDARG.F`; permissive IERS software licence, which grants derived works with
//! attribution). This is NOT IERS Conventions software and is neither distributed nor endorsed
//! by the IERS. The coefficient tables live in the generated `crate::subdaily_tables`; both
//! the tables and this re-implementation are certified against every routine's official test
//! case, in `derivations/eop_subdaily.py` and in the tests below.

// The IERS constants carry more digits than f64 holds; keep them verbatim from the routines.
#![allow(
    clippy::excessive_precision,
    clippy::doc_markdown,
    clippy::many_single_char_names
)]

use spacelab_core::Angle;

use crate::subdaily_tables::{CNMTX_LINES, ORTHOW, PMSDNUT2, SP, UTLIBR};

/// Arcseconds to radians (IERS `DAS2R`).
const DAS2R: f64 = 4.848_136_811_095_359_935_899_141e-6;
/// Arcseconds in a full circle.
const TURNAS: f64 = 1_296_000.0;
const TWOPI: f64 = core::f64::consts::TAU;
const PI: f64 = core::f64::consts::PI;
/// Modified Julian Date of J2000.
const RMJD0: f64 = 51544.5;
/// Radians to seconds of time.
const RAD2SEC: f64 = 86400.0 / TWOPI;

/// The diurnal/semidiurnal correction to add to the interpolated Earth-orientation parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubdailyCorrection {
    /// Correction to the pole `x_p`.
    pub dxp: Angle,
    /// Correction to the pole `y_p`.
    pub dyp: Angle,
    /// Correction to `UT1 − UTC`, in seconds.
    pub ddut1_seconds: f64,
    /// Correction to the excess length of day, in seconds.
    pub dlod_seconds: f64,
}

/// Delaunay fundamental arguments `(l, l', F, D, Ω)`, radians (IERS `FUNDARG`, Simon et al. 1994).
fn fundarg(t: f64) -> [f64; 5] {
    let poly = |a: f64, b: f64, c: f64, d: f64, e: f64| {
        ((a + t * (b + t * (c + t * (d + t * e)))) % TURNAS) * DAS2R
    };
    [
        poly(
            485_868.249_036,
            1_717_915_923.217_8,
            31.879_2,
            0.051_635,
            -0.000_244_70,
        ),
        poly(
            1_287_104.793_05,
            129_596_581.048_1,
            -0.553_2,
            0.000_136,
            -0.000_011_49,
        ),
        poly(
            335_779.526_232,
            1_739_527_262.847_8,
            -12.751_2,
            -0.001_037,
            0.000_004_17,
        ),
        poly(
            1_072_260.703_69,
            1_602_961_601.209_0,
            -6.370_6,
            0.006_593,
            -0.000_031_69,
        ),
        poly(
            450_160.398_036,
            -6_962_890.543_1,
            7.472_2,
            0.007_702,
            -0.000_059_39,
        ),
    ]
}

/// The six arguments `[GMST+π, l, l', F, D, Ω]` at `rmjd`, as used by the libration models.
fn arguments(rmjd: f64) -> [f64; 6] {
    let t = (rmjd - RMJD0) / 36525.0;
    let gmst = (67_310.548_41
        + t * ((8_640_184.812_866 + 3_155_760_000.0) + t * (0.093_104 + t * -0.000_006_2)))
        % 86400.0;
    let [l, lp, f, d, om] = fundarg(t);
    let arg1 = (gmst / RAD2SEC + PI) % TWOPI;
    [arg1, l, lp, f, d, om]
}

/// Angular argument `Σ multiplier·arg`, reduced.
fn angle(mult: [i8; 6], arg: &[f64; 6]) -> f64 {
    let mut a = 0.0;
    for i in 0..6 {
        a += f64::from(mult[i]) * arg[i];
    }
    a % TWOPI
}

/// Libration in polar motion `(Δx_p, Δy_p)`, microarcseconds (`PMSDNUT2`, Table 5.1a).
fn pmsdnut2_terms(rmjd: f64) -> (f64, f64) {
    let arg = arguments(rmjd);
    let (mut dx, mut dy) = (0.0, 0.0);
    for (mult, xs, xc, ys, yc) in PMSDNUT2 {
        let a = angle(mult, &arg);
        dx += xs * a.sin() + xc * a.cos();
        dy += ys * a.sin() + yc * a.cos();
    }
    (dx, dy)
}

/// Libration in UT1 (µs) and LOD (µs/day) (`UTLIBR`, Table 5.1b).
fn utlibr_terms(rmjd: f64) -> (f64, f64) {
    let arg = arguments(rmjd);
    let (mut dut1, mut dlod) = (0.0, 0.0);
    for (mult, us, uc, ls, lc) in UTLIBR {
        let a = angle(mult, &arg);
        dut1 += us * a.sin() + uc * a.cos();
        dlod += ls * a.sin() + lc * a.cos();
    }
    (dut1, dlod)
}

/// Partials `H(12)` of the tidal variation w.r.t. the orthoweights (`CNMTX`).
fn cnmtx(dmjd: f64) -> [f64; 12] {
    const D1960: f64 = 37076.5;
    // anm[m-1][k+1], bnm[m-1][k+1] for m ∈ {1,2}, k ∈ {-1,0,1}.
    let mut anm = [[0.0_f64; 3]; 2];
    let mut bnm = [[0.0_f64; 3]; 2];
    for (ki, k) in [-1.0, 0.0, 1.0].into_iter().enumerate() {
        let dt60 = (dmjd - k * 2.0) - D1960;
        for (j, (hs, phase, freq)) in CNMTX_LINES.into_iter().enumerate() {
            let m = usize::from(j >= 41); // 0 → diurnal (M=1), 1 → semidiurnal (M=2)
            // PINM = (N+M mod 2)·2π/4 with N=2: 2π/4 for the diurnal band, 0 for the semidiurnal.
            let pinm = if m == 0 { TWOPI / 4.0 } else { 0.0 };
            let alpha = (phase - pinm) % TWOPI + (freq * dt60) % TWOPI;
            anm[m][ki] += hs * alpha.cos();
            bnm[m][ki] -= hs * alpha.sin();
        }
    }
    let mut h = [0.0_f64; 12];
    for m in 0..2 {
        let ap = anm[m][2] + anm[m][0];
        let am = anm[m][2] - anm[m][0];
        let bp = bnm[m][2] + bnm[m][0];
        let bm = bnm[m][2] - bnm[m][0];
        let s = SP[m];
        let (a0, b0) = (anm[m][1], bnm[m][1]);
        let p = [
            s[0] * a0,
            s[1] * a0 - s[2] * ap,
            s[3] * a0 - s[4] * ap + s[5] * bm,
        ];
        let q = [
            s[0] * b0,
            s[1] * b0 - s[2] * bp,
            s[3] * b0 - s[4] * bp - s[5] * am,
        ];
        for k in 0..3 {
            h[m * 6 + k * 2] = p[k];
            h[m * 6 + k * 2 + 1] = q[k];
        }
    }
    h
}

/// Ocean-tidal `(Δx_p, Δy_p)` microarcseconds and `ΔUT1` microseconds (`ORTHO_EOP`).
fn ortho_eop(mjd: f64) -> (f64, f64, f64) {
    let h = cnmtx(mjd);
    let mut eop = [0.0_f64; 3];
    for (j, hj) in h.into_iter().enumerate() {
        for (k, e) in eop.iter_mut().enumerate() {
            *e += hj * ORTHOW[j][k];
        }
    }
    (eop[0], eop[1], eop[2])
}

/// The full IERS 2010 sub-daily correction at Modified Julian Date `mjd` (UTC).
///
/// Ocean tides (`ortho_eop`) plus libration (`pmsdnut2` in the pole, `utlibr` in UT1/LOD),
/// converted to `Angle` and seconds. Add to an interpolated [`crate::Eop`].
#[must_use]
pub fn subdaily_correction(mjd: f64) -> SubdailyCorrection {
    let (ox, oy, odut1_us) = ortho_eop(mjd);
    let (lx, ly) = pmsdnut2_terms(mjd);
    let (uut1_us, ulod_us) = utlibr_terms(mjd);
    SubdailyCorrection {
        dxp: Angle::from_arcseconds((ox + lx) * 1e-6),
        dyp: Angle::from_arcseconds((oy + ly) * 1e-6),
        ddut1_seconds: (odut1_us + uut1_us) * 1e-6,
        dlod_seconds: ulod_us * 1e-6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each IERS routine ships a test case in its header; these are those cases, so the Rust
    // port is certified against the same oracle as the generated tables. The libration
    // residual (~5e-7) is the benign GMST-path artifact documented in derivations/eop_subdaily.py.

    /// (G5) The "Test case" block of `FUNDARG.F`'s header, IERS Conventions (2010) chapter
    /// 5 software.
    #[test]
    fn fundarg_matches_the_iers_test_case() {
        let [l, lp, f, d, om] = fundarg(0.079_958_932_238_193_02);
        for (got, want) in [
            (l, 2.291_187_512_612_069_099),
            (lp, 6.212_931_111_003_726_414),
            (f, 3.658_025_792_050_572_989),
            (d, 4.554_139_562_402_433_228),
            (om, -0.516_737_921_723_180_448_9),
        ] {
            assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        }
    }

    /// (G5) The "Test case" block of `ORTHO_EOP.F`'s header, IERS Conventions (2010) chapter
    /// 8 software.
    #[test]
    fn ortho_eop_matches_the_iers_test_case() {
        let (dx, dy, dut1) = ortho_eop(47100.0);
        assert!((dx - -162.838_637_327_963_65).abs() < 1e-6, "dx {dx}");
        assert!((dy - 117.790_752_584_266_9).abs() < 1e-6, "dy {dy}");
        assert!((dut1 - -23.390_923_706_098_082).abs() < 1e-6, "dut1 {dut1}");
    }

    /// (G5) The "Test case" block of `PMSDNUT2.F`'s header, IERS Conventions (2010) chapter
    /// 5 software.
    #[test]
    fn pmsdnut2_matches_the_iers_test_case() {
        let (dx, dy) = pmsdnut2_terms(54335.0);
        assert!((dx - 24.831_442_382_733_65).abs() < 1e-3, "dx {dx}");
        assert!((dy - -14.092_406_920_418_377).abs() < 1e-3, "dy {dy}");
    }

    /// (G5) The "Test case" block of `UTLIBR.F`'s header, IERS Conventions (2010) chapter
    /// 5 software.
    #[test]
    fn utlibr_matches_the_iers_test_cases() {
        for (rmjd, wu, wl) in [
            (44239.1, 2.441_143_834_386_761_746, -14.789_712_473_494_495),
            (55227.4, -2.655_705_844_335_680_244, 27.394_458_265_998_47),
        ] {
            let (du, dl) = utlibr_terms(rmjd);
            assert!((du - wu).abs() < 1e-3, "dut1 {du} want {wu}");
            assert!((dl - wl).abs() < 1e-3, "dlod {dl} want {wl}");
        }
    }

    /// FUNDARG and the GMST polynomial, at epochs where their higher-order terms are separable.
    ///
    /// Every IERS-published test case for these models sits within a few decades of J2000, where
    /// the t², t³ and t⁴ terms are worth well under a microarcsecond: a sign flip in any of them,
    /// or a coefficient multiplied instead of added, changes nothing a published case can see. Ten
    /// centuries either side they are worth arcseconds. Reference values from
    /// `tools/gen_fundarg_oracle.mac` — Maxima at 60 digits, from the coefficients transcribed
    /// independently out of the IERS Conventions routines FUNDARG.F and ARG2.F, with the range
    /// reduction written to truncate toward zero the way Rust's `%` does.
    ///
    /// The bound is 1e-9 rad. Worst measured residual 1.5e-11 rad in the Delaunay arguments and
    /// 2.0e-10 in GMST — an ulp or two of the ~1e10-arcsecond and ~3e10-second polynomial values
    /// the reduction starts from, not a disagreement. The smallest mutation these epochs are meant
    /// to separate is worth 2e-7 rad, two hundred times the bound.
    #[test]
    fn the_argument_polynomials_hold_where_their_higher_order_terms_are_worth_something() {
        const CASES: [(f64, [f64; 5], f64); 4] = [
            (
                -10.0,
                [
                    -9.222_615_065_805_873e-1,
                    -6.160_824_437_144_564e0,
                    -1.266_478_443_089_597e-1,
                    -4.423_417_512_714_164e0,
                    4.644_745_503_565_451_7e-1,
                ],
                1.619_646_372_681_684e0,
            ),
            (
                -1.0,
                [
                    -1.115_183_659_436_996_4e0,
                    -2.655_231_639_918_746e-2,
                    -6.086_816_721_524_465e0,
                    -1.616_702_487_517_035_2e-1,
                    4.523_594_802_968_211e0,
                ],
                1.739_935_359_515_807_6e0,
            ),
            (
                0.3,
                [
                    2.552_451_184_187_611_3e-1,
                    6.235_087_221_592_769e0,
                    5.827_253_106_359_51e0,
                    5.549_858_524_301_061e0,
                    -1.661_486_020_925_838e0,
                ],
                4.898_993_813_439_823_7e0,
            ),
            (
                10.0,
                [
                    5.664_260_211_392_447e0,
                    6.074_036_564_672_733e0,
                    3.370_094_499_297_481e0,
                    2.247_799_894_833_134_3e0,
                    -2.375_541_973_345_720_6e0,
                ],
                1.888_444_888_573_827_5e0,
            ),
        ];

        for (t, want_fa, want_gmst) in CASES {
            let got = fundarg(t);
            for k in 0..5 {
                assert!(
                    (got[k] - want_fa[k]).abs() < 1e-9,
                    "t = {t}: fundarg[{k}] = {:.17e}, want {:.17e}",
                    got[k],
                    want_fa[k]
                );
            }
            let rmjd = RMJD0 + t * 36525.0;
            let got_gmst = arguments(rmjd)[0];
            assert!(
                (got_gmst - want_gmst).abs() < 1e-9,
                "t = {t}: GMST+π = {got_gmst:.17e}, want {want_gmst:.17e}"
            );
        }
    }

    /// The published correction is the sum of the three models, in the units it advertises.
    ///
    /// The existing test only bounds two of the four fields — an assertion that a microarcsecond
    /// quantity is below a milliarcsecond, which a sign flip, a dropped term and a scale factor all
    /// satisfy, and which says nothing at all about `dyp` or `dlod`. The three models are pinned
    /// against their own IERS cases above; what is left to check here is the plumbing, so this
    /// composes from them directly and requires each pair to actually disagree, or a sum and a
    /// difference would be the same number.
    #[test]
    fn the_correction_is_the_sum_of_the_three_models_converted_from_micro_units() {
        let mjd = 47_100.0;
        let (ox, oy, odut1_us) = ortho_eop(mjd);
        let (lx, ly) = pmsdnut2_terms(mjd);
        let (uut1_us, ulod_us) = utlibr_terms(mjd);

        // Non-vacuity: the two contributions to each field must differ, and the microarcsecond
        // scaling must be visible, or several of these mutations coincide with the truth.
        assert!(
            (ox - lx).abs() > 1.0 && (oy - ly).abs() > 1.0,
            "ocean ≈ libration"
        );
        assert!(
            (odut1_us - uut1_us).abs() > 1.0 && ulod_us.abs() > 1.0,
            "the UT1 terms are too close to separate"
        );

        let c = subdaily_correction(mjd);
        let close = |got: f64, want: f64, what: &str| {
            assert!((got - want).abs() < 1e-18, "{what}: {got:e} vs {want:e}");
        };
        close(c.dxp.arcseconds(), (ox + lx) * 1e-6, "Δxp, µas → arcsec");
        close(c.dyp.arcseconds(), (oy + ly) * 1e-6, "Δyp, µas → arcsec");
        close(c.ddut1_seconds, (odut1_us + uut1_us) * 1e-6, "ΔUT1, µs → s");
        close(c.dlod_seconds, ulod_us * 1e-6, "ΔLOD, µs → s");
    }

    #[test]
    fn subdaily_correction_is_the_signed_sum_of_the_models() {
        // At the ortho_eop test epoch the ocean term dominates; the correction must carry it.
        let c = subdaily_correction(47100.0);
        // dxp includes ocean −162.8 µas plus a libration term, all < 1 mas.
        assert!(
            c.dxp.arcseconds().abs() < 1e-3,
            "dxp {} as",
            c.dxp.arcseconds()
        );
        assert!(c.ddut1_seconds.abs() < 1e-4, "ddut1 {}", c.ddut1_seconds);
    }
}
