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
//! 2010, Technical Note 36, tables 5.3a/5.3b), machine-extracted and validated
//! against `pyerfa` (ERFA, BSD-3-Clause) over a spread of epochs to `< 1e-18` rad
//! and against the SOFA `t_nut00a` / `t_nut06a` self-test vectors — see
//! `derivations/nut00a_tables.py`. spacelab does not link SOFA or ERFA.
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
    #![allow(clippy::excessive_precision)] // verbatim SOFA golden-vector values

    use super::*;

    /// The SOFA self-test epoch, TT JD 2400000.5 + 53736.0 (Julian century 0.06).
    fn sofa_tt() -> Epoch<Tt> {
        Epoch::from_days_since_j2000(2191.5)
    }

    /// `Δψ, Δε` reproduce the SOFA `t_nut00a` golden vector. The tolerance is SOFA's
    /// own (`1e-13`); the series in fact matches to a few `1e-19` (see the
    /// derivation), the gap being the rounding of the printed golden constants.
    #[test]
    fn nutation_00a_matches_sofa_golden() {
        let (dpsi, deps) = nutation_00a(sofa_tt());
        let gdpsi = -9.630_909_107_115_518_431e-6;
        let gdeps = 4.063_239_174_001_678_710e-5;
        assert!(
            (dpsi - gdpsi).abs() < 1e-13,
            "dpsi: {dpsi:.19e} want {gdpsi:.19e}"
        );
        assert!(
            (deps - gdeps).abs() < 1e-13,
            "deps: {deps:.19e} want {gdeps:.19e}"
        );
    }

    /// `Δψ, Δε` reproduce the SOFA `t_nut06a` golden vector after the P03 rescale.
    #[test]
    fn nutation_06a_matches_sofa_golden() {
        let (dpsi, deps) = nutation_06a(sofa_tt());
        let gdpsi = -9.630_912_025_820_308_797e-6;
        let gdeps = 4.063_238_496_887_249_798e-5;
        assert!(
            (dpsi - gdpsi).abs() < 1e-13,
            "dpsi: {dpsi:.19e} want {gdpsi:.19e}"
        );
        assert!(
            (deps - gdeps).abs() < 1e-13,
            "deps: {deps:.19e} want {gdeps:.19e}"
        );
    }

    /// Both series against `pyerfa` over four centuries.
    ///
    /// The two golden vectors above stand at one epoch, and one epoch cannot separate a
    /// polynomial's coefficients from each other. `l′` and `D` are quartics in `t`
    /// ([`poly_arcsec`]); at `t = 0.06` their cubic and quartic terms are seven orders below
    /// the linear one, so corrupting the tail of the Horner nesting moves `Δψ` by ~1e-15 rad
    /// — a tenth of SOFA's own tolerance. `cargo mutants` finds five such corruptions the two
    /// golden vectors cannot see. Away from J2000 the same corruption is 1e-9 rad and larger.
    ///
    /// The agreement over these twenty values is exact to the bit — same series, same
    /// summation order, same `sin`/`cos` — so the tolerance is in ulps. The two are there for
    /// a future toolchain that contracts a multiply-add, not for an error the model can make.
    #[test]
    fn both_nutation_series_match_pyerfa_across_four_centuries() {
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
        for (days, psi_00a, eps_00a, psi_06a, eps_06a) in cases {
            let tt = Epoch::<Tt>::from_days_since_j2000(days);
            for (label, (dpsi, deps), want_psi, want_eps) in [
                ("nut00a", nutation_00a(tt), psi_00a, eps_00a),
                ("nut06a", nutation_06a(tt), psi_06a, eps_06a),
            ] {
                spacelab_testkit::assert_within_ulps(
                    dpsi,
                    want_psi,
                    2,
                    &format!("{label} dpsi at day {days}"),
                );
                spacelab_testkit::assert_within_ulps(
                    deps,
                    want_eps,
                    2,
                    &format!("{label} deps at day {days}"),
                );
            }
        }
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
