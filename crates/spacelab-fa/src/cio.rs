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
//! The `X, Y, s` series coefficients (`crate::cio_tables`) are machine-extracted by
//! `derivations/cio06_tables.py` and held term by term to the IERS Conventions (2010)
//! Tables 5.2a, 5.2b and 5.2d by `derivations/cio_iers.mac`;
//! `derivations/cio_nutation_series.mac` sums them in 40-digit arithmetic and writes
//! the values the tests hold [`cip_xy`] and [`cio_locator_s`] to. The ERA constants
//! are the IAU 2000 values (Capitaine et al. 2000);
//! `derivations/fundamental_arguments.mac` holds them to the published decimals and
//! writes the exact angles the ERA test holds [`era`] to. `pyerfa` and the SOFA
//! `t_xy06`, `t_s06` and `t_era00` vectors are comparators.

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
    // Exact values to 25 digits as the derivation prints them, and verbatim SOFA values.
    #![allow(clippy::excessive_precision, clippy::unreadable_literal)]

    use super::*;

    /// TT epoch (days since J2000) whose Julian-century value is exactly 0.06 — the
    /// SOFA self-test epoch, TT JD 2400000.5 + 53736.0.
    fn sofa_tt() -> Epoch<Tt> {
        Epoch::from_days_since_j2000(2191.5)
    }

    /// TT days since J2000 at which the series are held: `t = 0.06` (the SOFA self-test epoch), a
    /// century and half a century either side of J2000, and two centuries out — the epochs of
    /// [`crate::fundamental`]'s arguments.
    const SERIES_DAYS: [f64; 6] = [2191.5, -36_525.0, -18_262.5, 18_262.5, 36_525.0, 73_050.0];

    /// `X, Y` at [`SERIES_DAYS`]: eq (5.16) summed in 40-digit arithmetic over the IERS (2010)
    /// Tables 5.2a and 5.2b, to 25 digits (`derivations/cio_nutation_series.mac` reads this table
    /// back).
    const XY_EXACT: [[f64; 2]; 6] = [
        [5.791308486706010461598920e-4, 4.020579816732956051573133e-5],
        [
            -9.684090413737693249881655e-3,
            -1.189116481842916528410902e-4,
        ],
        [
            -4.865602325789341414289917e-3,
            1.314839585232979065839028e-5,
        ],
        [
            4.886533763528366947797060e-3,
            -5.341831990202507094031032e-5,
        ],
        [
            9.720704461729240764099680e-3,
            -6.730586996167223499345463e-5,
        ],
        [
            1.943690565478396271085283e-2,
            -4.742699904065940052508769e-4,
        ],
    ];

    /// `s` at [`SERIES_DAYS`] for the `X, Y` the doubles of [`XY_EXACT`] give: eq (5.18) summed over
    /// Table 5.2d likewise.
    const S_EXACT: [f64; 6] = [
        -1.220032213076461748658837e-8,
        -2.336591912459084400110537e-7,
        6.443279169766965442454561e-8,
        1.058366160190405539193861e-7,
        -4.805119345337526002948732e-9,
        1.832705450102923244183486e-6,
    ];

    /// The worst `X, Y` miss measured is 1.041e-17 rad, `X` two centuries out: three ulps of its
    /// 0.019 rad, inside what the arguments' own 1.6e-12 rad miss there makes of the 6.8″ 18.6-year
    /// term. The bound is 3.8 times that.
    const XY_BOUND: f64 = 4.0e-17;

    /// The worst `s` miss measured is 4.235e-21 rad, two centuries out: the arguments' 1.6e-12 rad
    /// miss there times the series' largest amplitude, 2640.73 µas. The bound is 3.5 times that.
    const S_BOUND: f64 = 1.5e-20;

    /// `X, Y` are their series summed exactly. (G3)
    ///
    /// One epoch near J2000 cannot see the series' high powers of `t`: at `t = 0.06` a `t⁴` Poisson
    /// term is a hundred-thousandth of its `t⁰` sibling. A century out and two centuries out they
    /// carry their own weight, and the epochs those sit at are the ones [`crate::fundamental`]'s
    /// arguments are held at.
    #[test]
    fn the_cip_is_its_series_summed_exactly() {
        let mut worst = (0.0_f64, 0.0_f64);
        for (days, want) in SERIES_DAYS.into_iter().zip(XY_EXACT) {
            let (x, y) = cip_xy(Epoch::from_days_since_j2000(days));
            for miss in [(x - want[0]).abs(), (y - want[1]).abs()] {
                if miss > worst.0 || miss.is_nan() {
                    worst = (miss, days);
                }
            }
        }
        assert!(worst.0 < XY_BOUND, "{:.3e} at day {}", worst.0, worst.1);
    }

    /// `s` is its series summed exactly, for given `X, Y`. (G3)
    #[test]
    fn the_cio_locator_is_its_series_summed_exactly() {
        let mut worst = (0.0_f64, 0.0_f64);
        for ((days, xy), want) in SERIES_DAYS.into_iter().zip(XY_EXACT).zip(S_EXACT) {
            let s = cio_locator_s(Epoch::from_days_since_j2000(days), xy[0], xy[1]);
            let miss = (s - want).abs();
            if miss > worst.0 || miss.is_nan() {
                worst = (miss, days);
            }
        }
        assert!(worst.0 < S_BOUND, "{:.3e} at day {}", worst.0, worst.1);
    }

    /// The distances to the SOFA `t_xy06` golden vector.
    #[test]
    #[ignore = "comparator: the distance to SOFA, not a verdict"]
    fn cip_xy_distance_to_sofa() {
        let (x, y) = cip_xy(sofa_tt());
        comparison_row(
            "sofa",
            "the CIP X, Y at the t_xy06 epoch",
            &[
                (x - 0.579_130_848_670_601_097_5e-3).abs(),
                (y - 0.402_057_981_673_295_814_1e-4).abs(),
            ],
        );
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

    /// The distance to the SOFA `t_s06` golden vector, for its reference `X, Y`.
    #[test]
    #[ignore = "comparator: the distance to SOFA, not a verdict"]
    fn cio_locator_distance_to_sofa() {
        let x = 0.579_130_848_670_601_100_0e-3;
        let y = 0.402_057_981_673_296_121_9e-4;
        let s = cio_locator_s(sofa_tt(), x, y);
        comparison_row(
            "sofa",
            "the CIO locator s at the t_s06 epoch",
            &[(s - -0.122_003_221_307_646_311_7e-7).abs()],
        );
    }

    /// UT1 days since J2000: the SOFA `t_era00` epoch (JD 2400000.5 + 54388.0), and five more whose
    /// day fractions are a quarter, an eighth and none at all.
    const ERA_DAYS: [f64; 6] = [2843.5, 2843.25, -1234.125, 18262.375, 0.0, -0.75];

    /// The ERA at [`ERA_DAYS`]: Capitaine et al. (2000) in exact arithmetic, reduced to `[0, 2π)`, to
    /// 25 digits (`derivations/fundamental_arguments.mac` reads this table back).
    const ERA_EXACT: [f64; 6] = [
        4.022837240028158101781625e-1,
        5.110372159493694735572060e0,
        1.729479102667400704953986e0,
        9.613592572988905425073335e-1,
        4.894961212823756883076502e0,
        1.696705977576342284823349e-1,
    ];

    /// The worst miss measured is 9.659e-15 rad, at day 18262.375: 0.22 of the 4.46e-14 rad that one
    /// ulp of its ~51-turn sum is. The bound is 3.1 times that.
    const ERA_BOUND: f64 = 3.0e-14;

    /// The ERA is its linear function of UT1, reduced to one turn. (G3)
    ///
    /// The SOFA epoch sits at UT1 midnight-plus-half-a-day, and a half turn of `frac` is exactly
    /// the half of a turn that the reduction to `[0, 2π)` throws away: negating `frac` there moves
    /// the angle by a whole turn and changes nothing. One epoch cannot separate the two terms
    /// `frac` appears in — the integer-stripped unit coefficient and the small linear one — so the
    /// other five carry fractions of a quarter, an eighth and none at all; mistaking the sign of a
    /// quarter-day fraction is 1.6 rad.
    #[test]
    fn era_is_its_linear_function_of_ut1_reduced_to_one_turn() {
        let mut worst = (0.0_f64, 0.0_f64);
        for (days, want) in ERA_DAYS.into_iter().zip(ERA_EXACT) {
            let miss = (era(Epoch::<Ut1>::from_days_since_j2000(days)).radians() - want).abs();
            if miss > worst.0 || miss.is_nan() {
                worst = (miss, days);
            }
        }
        assert!(worst.0 < ERA_BOUND, "{:.3e} at day {}", worst.0, worst.1);
    }

    /// The distance to the SOFA `t_era00` golden vector at UT1 JD 2400000.5 + 54388.0.
    #[test]
    #[ignore = "comparator: the distance to SOFA, not a verdict"]
    fn era_distance_to_sofa() {
        let got = era(Epoch::<Ut1>::from_days_since_j2000(2843.5)).radians();
        comparison_row(
            "sofa",
            "the ERA at the t_era00 epoch",
            &[(got - 0.402_283_724_002_815_810_2).abs()],
        );
    }

    /// The distances to `erfa.era00` at the five day fractions other than a half day.
    ///
    /// `era00` multiplies the whole turn count by 2π and reduces the radian value, so its answer
    /// carries the ulp of a ~320 rad quantity, while reducing in turns first — what [`era`] does —
    /// carries the ulp of ~51.
    #[test]
    #[ignore = "comparator: the distance to pyerfa, not a verdict"]
    fn era_distance_to_pyerfa_at_other_day_fractions() {
        let cases = [
            (2843.25_f64, 5.110_372_159_493_699_94e0),
            (-1234.125, 1.729_479_102_667_401_24e0),
            (18262.375, 9.613_592_572_989_020_37e-1),
            (0.0, 4.894_961_212_823_756_29e0),
            (-0.75, 1.696_705_977_576_341_28e-1),
        ];
        let distances = cases.map(|(days, want)| {
            (era(Epoch::<Ut1>::from_days_since_j2000(days)).radians() - want).abs()
        });
        comparison_row(
            "pyerfa",
            "the ERA at five day fractions other than a half day",
            &distances,
        );
    }

    /// One `comparison | ...` row of `tools/compare.py`, in radians.
    fn comparison_row(code: &str, quantity: &str, distances: &[f64]) {
        let max = distances.iter().copied().fold(0.0, f64::max);
        #[expect(clippy::cast_precision_loss, reason = "a count of a few distances")]
        let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
        println!(
            "comparison | {code} | not recorded | {quantity} | rad | {} | {max:.3e} | {rms:.3e}",
            distances.len()
        );
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
