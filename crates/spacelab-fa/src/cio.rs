//! Celestial Intermediate Origin (CIO) based Earth orientation, IAU 2006/2000A.
//!
//! The three quantities that place the Celestial Intermediate Pole and the CIO,
//! and turn the Earth beneath them:
//!
//! * [`cip_xy`] — the CIP coordinates `X, Y` in the GCRS, from the IAU 2006/2000A
//!   series (frame bias + precession + nutation), IERS Conventions 2010 tables
//!   5.2a/5.2b.
//! * [`cio_locator_s`] — the CIO locator `s`, positioning the CIO on the CIP
//!   equator, table 5.2c/5.2d.
//! * [`era`] — the Earth Rotation Angle, the linear IAU 2000 function of UT1.
//!
//! Together with polar motion these give the full GCRF→ITRF orientation; the frame
//! composition lives in `spacelab-frames`, which consumes these three.
//!
//! # Provenance
//!
//! The `X, Y, s` series coefficients (`crate::cio_tables`) are machine-extracted
//! from the IERS-published model and validated against `pyerfa` and the SOFA
//! self-test vectors to `< 5e-18` rad — see `derivations/cio06_tables.py`. The ERA
//! constants are the IAU 2000 values (Capitaine et al. 2000). Every
//! function here is cross-checked against `pyerfa` in the tests below.

use spacelab_core::{
    Angle,
    time::{Epoch, Tt, Ut1},
};

use crate::cio_tables::{
    S_COS, S_MULT, S_POLY, S_PT, S_SIN, XY_AMP, XY_CODE, XY_MULT, XY_OFFSET, XYP,
};
use crate::fundamental::{
    DAS2R, fad03, fae03, faf03, faju03, fal03, falp03, fama03, fame03, fane03, faom03, fapa03,
    fasa03, faur03, fave03,
};

/// The 14 fundamental arguments in the order the `X, Y` series indexes them.
fn fa14(t: f64) -> [f64; 14] {
    [
        fal03(t),
        falp03(t),
        faf03(t),
        fad03(t),
        faom03(t),
        fame03(t),
        fave03(t),
        fae03(t),
        fama03(t),
        faju03(t),
        fasa03(t),
        faur03(t),
        fane03(t),
        fapa03(t),
    ]
}

/// The 8 fundamental arguments in the order the `s` series indexes them
/// (`l, l', F, D, Ω, Venus, Earth, pA` — note Venus and Earth, not Mercury).
fn fa8(t: f64) -> [f64; 8] {
    [
        fal03(t),
        falp03(t),
        faf03(t),
        fad03(t),
        faom03(t),
        fave03(t),
        fae03(t),
        fapa03(t),
    ]
}

/// CIP coordinates `X, Y` (radians) in the GCRS, IAU 2006/2000A series.
///
/// Evaluates the polynomial part (arcseconds) plus the periodic series (amplitudes
/// in microarcseconds; `sin`/`cos` computed once per frequency), IERS Conventions
/// 2010 Eq. 5.16.
#[allow(clippy::many_single_char_names)] // IERS notation: t, x, y, s, c
#[must_use]
pub fn cip_xy(tt: Epoch<Tt>) -> (f64, f64) {
    let t = tt.julian_centuries();
    let fa = fa14(t);

    let mut pt = [1.0_f64; 6];
    for k in 1..6 {
        pt[k] = pt[k - 1] * t;
    }

    // Polynomial part (arcsec), high power first.
    let mut xpr = 0.0;
    let mut ypr = 0.0;
    for j in (0..6).rev() {
        xpr += XYP[0][j] * pt[j];
        ypr += XYP[1][j] * pt[j];
    }

    // Periodic part (microarcsec accumulators).
    let mut xacc = 0.0;
    let mut yacc = 0.0;
    for p in 0..1309 {
        let mut arg = 0.0;
        for k in 0..14 {
            let m = XY_MULT[p][k];
            if m != 0 {
                arg += f64::from(m) * fa[k];
            }
        }
        let s = libm::sin(arg);
        let c = libm::cos(arg);
        for idx in XY_OFFSET[p] as usize..XY_OFFSET[p + 1] as usize {
            let code = XY_CODE[idx];
            let trig = if code & 0b10 == 0 { s } else { c };
            let val = XY_AMP[idx] * trig * pt[(code >> 2) as usize];
            if code & 1 == 0 {
                xacc += val;
            } else {
                yacc += val;
            }
        }
    }

    let x = DAS2R * (xpr + xacc / 1e6);
    let y = DAS2R * (ypr + yacc / 1e6);
    (x, y)
}

/// CIO locator `s` (radians) given the CIP coordinates, IAU 2006/2000A.
///
/// `s` positions the CIO on the CIP equator; IERS Conventions 2010 Eq. 5.18. The
/// `− x·y/2` term is the exact geometric part, the series the small remainder.
#[allow(clippy::many_single_char_names)] // IERS notation: t, x, y, w, s
#[must_use]
pub fn cio_locator_s(tt: Epoch<Tt>, x: f64, y: f64) -> f64 {
    let t = tt.julian_centuries();
    let fa = fa8(t);

    let mut w = S_POLY;
    for i in 0..66 {
        let mut arg = 0.0;
        for k in 0..8 {
            let m = S_MULT[i][k];
            if m != 0 {
                arg += f64::from(m) * fa[k];
            }
        }
        w[S_PT[i] as usize] += S_SIN[i] * libm::sin(arg) + S_COS[i] * libm::cos(arg);
    }

    let poly = (w[0] + (w[1] + (w[2] + (w[3] + (w[4] + w[5] * t) * t) * t) * t) * t) * DAS2R;
    poly - x * y / 2.0
}

/// Earth Rotation Angle (IAU 2000) as a function of UT1.
///
/// `ERA(Tu) = 2π (0.779_057_273_264_0 + 1.002_737_811_911_354_48 · Tu)` with `Tu`
/// the number of UT1 days since J2000.0 (Capitaine et al. 2000). The
/// unit coefficient on `Tu` contributes only whole turns, so the exact day fraction
/// carries it and the large day count never enters the reduced angle.
#[must_use]
pub fn era(ut1: Epoch<Ut1>) -> Angle {
    let (day, frac) = ut1.day_and_fraction();
    #[allow(clippy::cast_precision_loss)] // |day| ≪ 2^53 for any epoch of interest
    let tu = day as f64 + frac;
    // 0.779… + frac carries the integer-stripped unit term; the small linear term
    // uses the full day count.
    let turns = 0.779_057_273_264_0 + frac + 0.002_737_811_911_354_48 * tu;
    Angle::from_radians(reduce_turns_to_radians(turns))
}

/// Reduce a value in turns to `[0, 2π)` radians.
fn reduce_turns_to_radians(turns: f64) -> f64 {
    let mut f = turns - libm::floor(turns);
    if f >= 1.0 {
        f -= 1.0;
    } else if f < 0.0 {
        f += 1.0;
    }
    debug_assert!((0.0..1.0).contains(&f));
    f * Angle::TAU
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)] // verbatim SOFA golden-vector values

    use super::*;

    /// TT epoch (days since J2000) whose Julian-century value is exactly 0.06 — the
    /// SOFA self-test epoch, TT JD 2400000.5 + 53736.0.
    fn sofa_tt() -> Epoch<Tt> {
        Epoch::from_days_since_j2000(2191.5)
    }

    /// `X, Y` reproduce the SOFA `t_xy06` golden vector (radians). The series and the
    /// coefficient tables are exercised end to end.
    #[test]
    fn cip_xy_matches_sofa_golden() {
        let (x, y) = cip_xy(sofa_tt());
        let gx = 0.579_130_848_670_601_097_5e-3;
        let gy = 0.402_057_981_673_295_814_1e-4;
        assert!((x - gx).abs() < 1e-15, "X: got {x:.19e} want {gx:.19e}");
        assert!((y - gy).abs() < 1e-15, "Y: got {y:.19e} want {gy:.19e}");
    }

    /// The `X, Y` series carries the **full** IAU 2000A nutation, not a truncation. (S1)
    ///
    /// The classical (equinox-based) route builds `X, Y` from frame bias +
    /// precession + the complete 678+687-term IAU 2000A nutation ([`crate::nutation`])
    /// and reads them off the resulting pole; ERFA exposes it as `xys06a`. The `X, Y`
    /// series evaluated by [`cip_xy`] is an independent, direct-series formulation of
    /// the same IAU 2006/2000A model. That the two agree to sub-microarcsecond here
    /// (`|ΔX| ≈ 3.9e-13`, `|ΔY| ≈ 2.8e-12` rad — well under `5e-11`) can only hold if
    /// the series encodes the full nutation; a truncated nutation would diverge by
    /// orders of magnitude more. Reference `X, Y` from `erfa.xys06a` at the SOFA
    /// epoch — see `derivations/nut00a_tables.py`.
    #[test]
    fn cip_xy_series_carries_full_2000a_nutation() {
        let (x, y) = cip_xy(sofa_tt());
        let xys06a_x = 5.791_308_482_835_291_066_0e-4;
        let xys06a_y = 4.020_580_099_467_485_624_1e-5;
        let dx = (x - xys06a_x).abs();
        let dy = (y - xys06a_y).abs();
        assert!(dx < 5e-11, "X vs xys06a: {dx:.3e}");
        assert!(dy < 5e-11, "Y vs xys06a: {dy:.3e}");
    }

    /// `s` reproduces the SOFA `t_s06` golden vector for its reference `X, Y`.
    #[test]
    fn cio_locator_matches_sofa_golden() {
        let x = 0.579_130_848_670_601_100_0e-3;
        let y = 0.402_057_981_673_296_121_9e-4;
        let s = cio_locator_s(sofa_tt(), x, y);
        let gs = -0.122_003_221_307_646_311_7e-7;
        assert!((s - gs).abs() < 1e-20, "s: got {s:.19e} want {gs:.19e}");
    }

    /// ERA reproduces the SOFA `t_era00` golden vector at UT1 JD 2400000.5 + 54388.0.
    #[test]
    fn era_matches_sofa_golden() {
        let ut1 = Epoch::<Ut1>::from_days_since_j2000(2843.5);
        let got = era(ut1).radians();
        let want = 0.402_283_724_002_815_810_2;
        assert!(
            (got - want).abs() < 1e-12,
            "ERA: got {got:.19e} want {want:.19e}"
        );
    }

    /// ERA at day fractions other than a half day.
    ///
    /// The golden vector above sits at UT1 midnight-plus-half-a-day, and a half turn of
    /// `frac` is exactly the half of a turn that the reduction to `[0, 2π)` throws away:
    /// negating `frac` there moves the angle by a whole turn and changes nothing. One
    /// epoch cannot separate the two terms `frac` appears in — the integer-stripped unit
    /// coefficient and the small linear one — so these five carry fractions of a quarter,
    /// an eighth and none at all. Reference from `erfa.era00`.
    ///
    /// The tolerance is `erfa`'s conditioning, not ours: `era00` multiplies the whole turn
    /// count by 2π and reduces the radian value, so its answer carries the ulp of a ~320 rad
    /// quantity, while reducing in turns first — what [`era`] does — carries the ulp of ~51.
    /// The worst residual over these five is 2.12e-14 rad, at the epoch furthest from J2000;
    /// mistaking the sign of a quarter-day fraction is 1.6 rad.
    #[test]
    fn era_matches_pyerfa_at_other_day_fractions() {
        let cases = [
            (2843.25_f64, 5.110_372_159_493_699_94e0),
            (-1234.125, 1.729_479_102_667_401_24e0),
            (18262.375, 9.613_592_572_989_020_37e-1),
            (0.0, 4.894_961_212_823_756_29e0),
            (-0.75, 1.696_705_977_576_341_28e-1),
        ];
        for (days, want) in cases {
            let got = era(Epoch::<Ut1>::from_days_since_j2000(days)).radians();
            assert!(
                (got - want).abs() < 5e-14,
                "ERA at day {days}: got {got:.19e} want {want:.19e}"
            );
        }
    }

    /// The reduction is total: every finite input leaves an angle in `[0, 2π)`.
    ///
    /// `turns - floor(turns)` is exact in `[0, 1)` for almost every argument, but for a
    /// tiny negative one it is `1 - ε` rounded, which is `1.0` — the case the correction
    /// after the subtraction exists for. No UT1 reaches it through [`era`]; that is why it
    /// is tested here rather than through the public function.
    #[test]
    #[allow(clippy::float_cmp)] // exactly zero, and being exactly it is the assertion
    fn the_turns_reduction_leaves_no_angle_outside_one_turn() {
        for turns in [
            -1e-20,
            -1e-17,
            -f64::MIN_POSITIVE,
            0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1e9 + 0.25,
            -1e9 - 0.25,
        ] {
            let a = reduce_turns_to_radians(turns);
            assert!(
                (0.0..Angle::TAU).contains(&a),
                "{turns:e} turns reduced to {a}"
            );
        }
        // A whole number of turns is no angle at all, not a whole turn.
        assert_eq!(reduce_turns_to_radians(0.0), 0.0);
        assert_eq!(reduce_turns_to_radians(-3.0), 0.0);
    }

    /// The IAU 2006/2000A `X, Y, s` series coefficients, sealed. (S1)
    ///
    /// A second parse of our own generated file, not a document's vector.
    ///
    /// The SOFA golden vectors above agree to `1e-20` rad, but that agreement is carried by the
    /// first few of 4755 amplitudes; a sign deleted deep in the series moves `X` far below it.
    /// `cargo mutants` finds 36 such deletions and no test here sees one. The constant comes from
    /// `derivations/table_seals.py`, which reads the generated file with Python's float parser — a
    /// different parse of the same text than rustc's.
    #[test]
    fn the_cip_and_cio_series_are_the_ones_the_generator_wrote() {
        let mut flat: Vec<f64> = Vec::new();
        flat.extend(XYP.iter().flatten().copied());
        flat.extend(XY_MULT.iter().flatten().map(|&m| f64::from(m)));
        flat.extend(XY_OFFSET.iter().map(|&o| f64::from(o)));
        flat.extend(XY_CODE.iter().map(|&c| f64::from(c)));
        flat.extend(XY_AMP);
        flat.extend(S_POLY);
        flat.extend(S_MULT.iter().flatten().map(|&m| f64::from(m)));
        flat.extend(S_PT.iter().map(|&p| f64::from(p)));
        flat.extend(S_SIN);
        flat.extend(S_COS);
        assert_eq!(
            flat.len(),
            2 * 6 + 1309 * 14 + 1310 + 2 * 4755 + 6 + 66 * 11,
            "series shape moved"
        );
        assert_eq!(
            spacelab_testkit::fnv1a_f64(flat),
            0x88c8_2e37_40b7_f421,
            "IAU 2006/2000A X, Y, s coefficients moved"
        );
    }
}
