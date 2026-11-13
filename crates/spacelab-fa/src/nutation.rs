//! Equinox-based nutation, IAU 2000A / 2006A.
//!
//! The classical nutation angles — nutation in longitude `Δψ` and in obliquity
//! `Δε`, referred to the ecliptic and mean equinox of date:
//!
//! * [`nutation_00a`] — the IAU 2000A (MHB2000) series: a 678-term luni-solar
//!   series plus a 687-term planetary series, free core nutation omitted.
//! * [`nutation_06a`] — the same angles rescaled by the small IAU 2006 (P03)
//!   precession-rate and secular-`J₂` factors.
//!
//! This is the equinox-based route. The CIO-based route ([`crate::cio`]) needs no
//! separate nutation angles: the `X, Y` series it evaluates already carries this
//! same full IAU 2000A nutation, folded together with frame bias and precession —
//! a property a test in [`crate::cio`] pins directly against the classical route.
//!
//! # Provenance
//!
//! The series coefficients (`crate::nut_tables`) are the IERS/IAU-published
//! MHB2000 model constants (Mathews, Herring & Buffett 2002; IERS Conventions
//! 2010, Technical Note 36, tables 5.3a/5.3b), machine-extracted by
//! `derivations/nut00a_tables.py` and held row for row to the IERS tables by
//! `derivations/nutation_iers.mac`. `derivations/cio_nutation_series.mac` sums both
//! series in 40-digit arithmetic, applies the IAU 2006 factors as their published
//! decimals, and writes the values the tests hold [`nutation_00a`] and
//! [`nutation_06a`] to. `pyerfa` (ERFA, BSD-3-Clause) and the SOFA `t_nut00a` /
//! `t_nut06a` self-test vectors are comparators; spacelab does not link SOFA or ERFA.
//!
//! The luni-solar and planetary blocks are summed with slightly different
//! Delaunay-type arguments, exactly as MHB2000 prescribes: the luni-solar `l′`, `D`
//! and the planetary `l, F, D, Ω` and Neptune longitude are the MHB2000 polynomials
//! carried in `crate::nut_tables`, distinct in their last digits from the IERS
//! 2003 arguments of [`crate::fundamental`]; the remaining arguments are shared.

use spacelab_core::{
    Angle,
    time::{Epoch, Tt},
};

use crate::fundamental::{
    DAS2R, fae03, faf03, faju03, fal03, fama03, fame03, faom03, fapa03, fasa03, faur03, fave03,
};
use crate::nut_tables::{
    LS_D, LS_ELP, PL_AD, PL_AF, PL_AL, PL_ANE, PL_AOM, XLS_COEF, XLS_MULT, XPL_COEF, XPL_MULT,
};

/// One 0.1-microarcsecond unit in radians — the unit of every tabulated coefficient.
const U2R: f64 = DAS2R / 1e7;
/// Arcseconds in one full turn, for reducing the MHB2000 Delaunay polynomials.
const TURNAS: f64 = 1_296_000.0;

/// Evaluate an arcsecond polynomial `c₀ + t(c₁ + t(c₂ + t(c₃ + t·c₄)))`, reduce it
/// modulo a full turn and convert to radians — the MHB2000 form for `l′` and `D`.
fn poly_arcsec(c: [f64; 5], t: f64) -> f64 {
    let a = c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * c[4])));
    libm::fmod(a, TURNAS) * DAS2R
}

/// Evaluate a linear argument `c₀ + c₁·t` (radians), reduced modulo `2π`.
fn linear_rad(c: [f64; 2], t: f64) -> f64 {
    libm::fmod(c[0] + c[1] * t, Angle::TAU)
}

/// The five luni-solar arguments in series order `(l, l′, F, D, Ω)`, radians.
///
/// `l, F, Ω` are the IERS 2003 arguments ([`crate::fundamental`]); `l′` and `D` are
/// the faithfully distinct MHB2000 polynomials.
fn luni_solar_args(t: f64) -> [f64; 5] {
    [
        fal03(t),
        poly_arcsec(LS_ELP, t),
        faf03(t),
        poly_arcsec(LS_D, t),
        faom03(t),
    ]
}

/// The thirteen planetary arguments in series order
/// `(l, F, D, Ω, Me, Ve, Ea, Ma, Ju, Sa, Ur, Ne, pA)`, radians.
///
/// `l, F, D, Ω` and the Neptune longitude are the MHB2000 arguments; Mercury
/// through Uranus and the accumulated precession `pA` are the IERS 2003 arguments.
fn planetary_args(t: f64) -> [f64; 13] {
    [
        linear_rad(PL_AL, t),
        linear_rad(PL_AF, t),
        linear_rad(PL_AD, t),
        linear_rad(PL_AOM, t),
        fame03(t),
        fave03(t),
        fae03(t),
        fama03(t),
        faju03(t),
        fasa03(t),
        faur03(t),
        linear_rad(PL_ANE, t),
        fapa03(t),
    ]
}

/// Nutation in longitude and obliquity `(Δψ, Δε)` (radians), IAU 2000A.
///
/// The MHB2000 model referred to the ecliptic and mean equinox of date, free core
/// nutation omitted (IERS Conventions 2010, §5.5). Each series is summed from its
/// smallest term upward, as the published algorithm prescribes.
#[allow(clippy::many_single_char_names)] // IERS notation: t, m, s (sin), c (cos)
#[must_use]
pub fn nutation_00a(tt: Epoch<Tt>) -> (f64, f64) {
    let t = tt.julian_centuries();

    // Luni-solar series.
    let ls = luni_solar_args(t);
    let mut dp = 0.0;
    let mut de = 0.0;
    for i in (0..XLS_MULT.len()).rev() {
        let mult = XLS_MULT[i];
        let mut arg = 0.0;
        for k in 0..5 {
            let m = mult[k];
            if m != 0 {
                arg += f64::from(m) * ls[k];
            }
        }
        let arg = libm::fmod(arg, Angle::TAU);
        let s = libm::sin(arg);
        let c = libm::cos(arg);
        let co = XLS_COEF[i];
        dp += (co[0] + co[1] * t) * s + co[2] * c;
        de += (co[3] + co[4] * t) * c + co[5] * s;
    }
    let dpsi_ls = dp * U2R;
    let deps_ls = de * U2R;

    // Planetary series.
    let pl = planetary_args(t);
    let mut dp = 0.0;
    let mut de = 0.0;
    for i in (0..XPL_MULT.len()).rev() {
        let mult = XPL_MULT[i];
        let mut arg = 0.0;
        for k in 0..13 {
            let m = mult[k];
            if m != 0 {
                arg += f64::from(m) * pl[k];
            }
        }
        let arg = libm::fmod(arg, Angle::TAU);
        let s = libm::sin(arg);
        let c = libm::cos(arg);
        let co = XPL_COEF[i];
        dp += f64::from(co[0]) * s + f64::from(co[1]) * c;
        de += f64::from(co[2]) * s + f64::from(co[3]) * c;
    }
    let dpsi_pl = dp * U2R;
    let deps_pl = de * U2R;

    (dpsi_ls + dpsi_pl, deps_ls + deps_pl)
}

/// Nutation `(Δψ, Δε)` (radians) with the IAU 2006 (P03) adjustment applied.
///
/// Rescales the IAU 2000A nutation by the small factors that account for the
/// change in precession rates and the secular variation of `J₂` (Wallace &
/// Capitaine 2006): with `f_{J2} = −2.7774×10⁻⁶ · t`,
/// `Δψ ← Δψ (1 + 0.4697×10⁻⁶ + f_{J2})` and `Δε ← Δε (1 + f_{J2})`.
#[must_use]
pub fn nutation_06a(tt: Epoch<Tt>) -> (f64, f64) {
    let t = tt.julian_centuries();
    let fj2 = -2.7774e-6 * t;
    let (dpsi, deps) = nutation_00a(tt);
    (dpsi + dpsi * (0.4697e-6 + fj2), deps + deps * fj2)
}

#[cfg(test)]
mod tests {
    // Exact values to 25 digits as the derivation prints them, and verbatim SOFA and pyerfa values.
    #![allow(clippy::excessive_precision, clippy::unreadable_literal)]

    use super::*;

    /// The SOFA self-test epoch, TT JD 2400000.5 + 53736.0 (Julian century 0.06).
    fn sofa_tt() -> Epoch<Tt> {
        Epoch::from_days_since_j2000(2191.5)
    }

    /// TT days since J2000 at which the series are held: `t = 0.06` (the SOFA self-test epoch), a
    /// century and half a century either side of J2000, and two centuries out — the epochs of
    /// [`crate::fundamental`]'s arguments.
    const SERIES_DAYS: [f64; 6] = [2191.5, -36_525.0, -18_262.5, 18_262.5, 36_525.0, 73_050.0];

    /// IAU 2000A `Δψ, Δε` and IAU 2006A `Δψ, Δε` at [`SERIES_DAYS`]: both series summed in 40-digit
    /// arithmetic, the 2006 factors applied exactly, to 25 digits
    /// (`derivations/cio_nutation_series.mac` reads this table back).
    const NUT_EXACT: [[f64; 4]; 6] = [
        [
            -9.630909107115593210319210e-6,
            4.063239174001676613905611e-5,
            -9.630912025820383576342485e-6,
            4.063238496887247701570239e-5,
        ],
        [
            8.409753134603515754101405e-5,
            -1.111631344198687425678782e-5,
            8.409780441912919125177410e-5,
            -1.111634431643582803113238e-5,
        ],
        [
            -1.601424665756253387386411e-5,
            4.035162305736643226550146e-5,
            -1.601427641843852228807706e-5,
            4.035167909366537203026595e-5,
        ],
        [
            7.355346965281273809044773e-5,
            -2.583921583470367497508375e-5,
            7.355340205717412715554143e-5,
            -2.583917995178464532209032e-5,
        ],
        [
            1.585987390484705170619881e-5,
            4.162326779427022219241577e-5,
            1.585983730501604149065759e-5,
            4.162315218980625038630065e-5,
        ],
        [
            5.432795503198357145486201e-5,
            -3.868461745385661471975393e-5,
            5.432767876889943831520281e-5,
            -3.868440256854358203703049e-5,
        ],
    ];

    /// The worst miss measured is 2.580e-17 rad, IAU 2000A `Δψ` two centuries out: inside what the
    /// arguments' own 1.6e-12 rad miss there makes of the 17.2″ `Ω` term. The bound is 3.9 times
    /// that.
    const NUT_BOUND: f64 = 1.0e-16;

    /// Both nutations are their series summed exactly. (G3)
    ///
    /// One epoch cannot separate a polynomial's coefficients from each other. `l′` and `D` are
    /// quartics in `t` ([`poly_arcsec`]); at `t = 0.06` their cubic and quartic terms are seven
    /// orders below the linear one, so corrupting the tail of the Horner nesting moves `Δψ` by
    /// ~1e-15 rad, and `cargo mutants` found five such corruptions one epoch cannot see. A century
    /// and two centuries out the same corruption is 1e-9 rad and larger.
    #[test]
    fn both_nutations_are_their_series_summed_exactly() {
        let mut worst = (0.0_f64, String::new());
        for (days, want) in SERIES_DAYS.into_iter().zip(NUT_EXACT) {
            let tt = Epoch::<Tt>::from_days_since_j2000(days);
            let (psi_00a, eps_00a) = nutation_00a(tt);
            let (psi_06a, eps_06a) = nutation_06a(tt);
            for (i, got) in [psi_00a, eps_00a, psi_06a, eps_06a].into_iter().enumerate() {
                let miss = (got - want[i]).abs();
                if miss > worst.0 || miss.is_nan() {
                    worst = (miss, format!("column {i} at day {days}"));
                }
            }
        }
        assert!(worst.0 < NUT_BOUND, "{:.3e} at {}", worst.0, worst.1);
    }

    /// The distances to the SOFA `t_nut00a` and `t_nut06a` golden vectors.
    #[test]
    #[ignore = "comparator: the distance to SOFA, not a verdict"]
    fn nutation_distance_to_sofa() {
        let (psi_00a, eps_00a) = nutation_00a(sofa_tt());
        let (psi_06a, eps_06a) = nutation_06a(sofa_tt());
        comparison_row(
            "sofa",
            "IAU 2000A and 2006A nutation at the t_nut00a and t_nut06a epoch",
            &[
                (psi_00a - -9.630_909_107_115_518_431e-6).abs(),
                (eps_00a - 4.063_239_174_001_678_710e-5).abs(),
                (psi_06a - -9.630_912_025_820_308_797e-6).abs(),
                (eps_06a - 4.063_238_496_887_249_798e-5).abs(),
            ],
        );
    }

    /// The distances to `erfa.nut00a` and `erfa.nut06a` at five epochs over four centuries.
    ///
    /// Same series, same summation order, same `sin`/`cos`: the agreement is to the bit.
    #[test]
    #[ignore = "comparator: the distance to pyerfa, not a verdict"]
    fn nutation_distance_to_pyerfa_across_four_centuries() {
        // (days since J2000 TT, nut00a Δψ, Δε, nut06a Δψ, Δε)
        let cases = [
            (
                -36_525.0_f64,
                8.409_753_134_604_065_75e-5,
                -1.111_631_344_198_608_41e-5,
                8.409_780_441_913_468_82e-5,
                -1.111_634_431_643_503_87e-5,
            ),
            (
                -18_262.5,
                -1.601_424_665_756_548_94e-5,
                4.035_162_305_736_600_22e-5,
                -1.601_427_641_844_147_69e-5,
                4.035_167_909_366_494_17e-5,
            ),
            (
                18_262.5,
                7.355_346_965_280_616_63e-5,
                -2.583_921_583_470_474_45e-5,
                7.355_340_205_716_755_30e-5,
                -2.583_917_995_178_571_59e-5,
            ),
            (
                36_525.0,
                1.585_987_390_484_146_85e-5,
                4.162_326_779_426_947_75e-5,
                1.585_983_730_501_045_69e-5,
                4.162_315_218_980_550_80e-5,
            ),
            (
                73_050.0,
                5.432_795_503_195_776_96e-5,
                -3.868_461_745_386_165_13e-5,
                5.432_767_876_887_363_74e-5,
                -3.868_440_256_854_861_66e-5,
            ),
        ];
        let distances: Vec<f64> = cases
            .iter()
            .flat_map(|&(days, psi_00a, eps_00a, psi_06a, eps_06a)| {
                let tt = Epoch::<Tt>::from_days_since_j2000(days);
                let (p0, e0) = nutation_00a(tt);
                let (p6, e6) = nutation_06a(tt);
                [
                    (p0 - psi_00a).abs(),
                    (e0 - eps_00a).abs(),
                    (p6 - psi_06a).abs(),
                    (e6 - eps_06a).abs(),
                ]
            })
            .collect();
        comparison_row(
            "pyerfa",
            "IAU 2000A and 2006A nutation at t = -1, -0.5, 0.5, 1, 2",
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

    /// The IAU 2000A series coefficients, sealed. (S1)
    ///
    /// A second parse of our own generated file, not a document's vector.
    ///
    /// The SOFA golden vectors above agree to `1e-13` rad, which the handful of largest terms
    /// dominate; deleting the sign of a coefficient far down the 1365-term series moves `Δψ` by
    /// orders of magnitude less than that, so `cargo mutants` finds 63 such sign deletions the
    /// whole suite catches none of. The constant comes from `derivations/table_seals.py`, which
    /// reads the generated file with Python's float parser — a different parse of the same text
    /// than rustc's.
    #[test]
    fn the_nutation_series_is_the_one_the_generator_wrote() {
        let mut flat: Vec<f64> = Vec::new();
        flat.extend(XLS_MULT.iter().flatten().map(|&m| f64::from(m)));
        flat.extend(XLS_COEF.iter().flatten().copied());
        flat.extend(XPL_MULT.iter().flatten().map(|&m| f64::from(m)));
        flat.extend(XPL_COEF.iter().flatten().map(|&m| f64::from(m)));
        flat.extend(LS_ELP);
        flat.extend(LS_D);
        for arg in [PL_AL, PL_AF, PL_AD, PL_AOM, PL_ANE] {
            flat.extend(arg);
        }
        assert_eq!(
            flat.len(),
            678 * 11 + 687 * 17 + 2 * 5 + 5 * 2,
            "series shape moved"
        );
        assert_eq!(
            spacelab_testkit::fnv1a_f64(flat),
            0x7636_099b_00f3_c8e5,
            "IAU 2000A nutation coefficients moved"
        );
    }
}
