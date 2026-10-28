//! Rotation of the Earth about its axis.
//!
//! Gate: `derivations/gmst82.mac` shows the coded GMST82 split differs from the printed polynomial
//! by whole days only, derives the sidereal rate, certifies the four coefficients' bit patterns as
//! the nearest doubles in exact rationals, and evaluates GMST in exact rationals at five epochs that
//! `derivations/gmst82.gp` re-derives from the printed polynomial. The module tests
//! `gmst_is_the_exact_rational_polynomial_reduced_to_a_turn` (G2) and
//! `coefficients_are_the_certified_nearest_f64` (G3) tie the code to it.

use spacelab_core::{
    Angle,
    time::{Epoch, SECONDS_PER_DAY, Tt, Ut1},
};

// GMST(IAU 1982) coefficients, seconds of time, from Aoki et al. (1982) / Vallado Eq. 3-47.
// Each is the nearest `f64` to its published decimal, certified to under half an ulp by a
// 50-digit mpmath evaluation in `derivations/gmst82.py`; the exact bit patterns are asserted
// in the tests, so a mistyped digit fails there rather than silently shifting GMST.

/// Constant term, seconds of time. GMST at J2000.0 = 18ʰ41ᵐ50ˢ.54841.
const C0: f64 = 67_310.548_41;
/// Linear term beyond the whole-day part, seconds per Julian century.
const C1: f64 = 8_640_184.812_866;
/// Quadratic term, seconds per Julian century squared.
const C2: f64 = 0.093_104;
/// Cubic term, seconds per Julian century cubed.
const C3: f64 = -6.2e-6;

/// Greenwich Mean Sidereal Time, IAU 1982 model.
///
/// # The equation
///
/// Aoki et al. (1982); Vallado, *Fundamentals of Astrodynamics and Applications*,
/// Eq. 3-47. With `T` the number of Julian centuries of UT1 elapsed since J2000
/// and `θ` in seconds of time:
///
/// ```text
/// θ = 67310.54841 + (876600ʰ + 8640184.812866ˢ)·T + 0.093104·T² − 6.2×10⁻⁶·T³
/// ```
///
/// # Why the linear term is not evaluated as written
///
/// The coefficient `876600ʰ` is `3_155_760_000` seconds per Julian century, and
///
/// ```text
/// 876600 × 3600 = 86400 × 36525     exactly
/// ```
///
/// so that term equals `86400·D`, where `D` is the number of days of UT1 since
/// J2000. Since the result is reduced modulo one day, and `D = day + frac` with
/// `day` an integer,
///
/// ```text
/// 86400·D  ≡  86400·frac   (mod 86400)
/// ```
///
/// Evaluating `3_155_760_000·T` directly would form a quantity of order 10⁹ whose
/// last bit is worth about 10⁻⁷ s, and then cancel almost all of it away. Using
/// `86400·frac` instead never forms it.
///
/// This is a statement about the *printed* equation, not a claim of novelty: ERFA
/// reaches an equivalent decomposition by absorbing the same term into a
/// fractional-day quantity. Anyone transcribing Eq. 3-47 literally, however, pays
/// the full price.
///
/// # Conditioning
///
/// The largest surviving intermediate is `8640184.812866·T`, of order 10⁶·T
/// seconds. Its unit in the last place is worth `ε·|8.64×10⁶·T|` seconds, so two
/// mathematically equivalent orderings of this polynomial may disagree by a few
/// times `1.4×10⁻¹³·|T|` radians. Five thousand years from J2000 that is about
/// 10⁻¹¹ rad. No implementation of this equation in `f64` can do better, and any
/// tolerance tighter than that is measuring the summation order, not the formula.
///
/// # Provenance
///
/// Derived from the published equation. Not transcribed from ERFA, SOFA, or any
/// other implementation. Cross-checked against `pyerfa` — see `derivations/gmst82.py`.
///
/// The four coefficients (`C0`…`C3`, module scope) are the published decimals of
/// Eq. 3-47, each certified by `derivations/gmst82.py` to be the nearest `f64` to
/// the exact value (50-digit mpmath, all within half an ulp). Their bit patterns
/// are asserted in the tests, so a mistyped digit fails there rather than quietly
/// moving the Earth.
#[must_use]
pub fn gmst82(t: Epoch<Ut1>) -> Angle {
    let (_day, frac) = t.day_and_fraction();
    let c = t.julian_centuries();

    // Horner in `c`, with the whole-day part already reduced.
    let seconds = C0 + SECONDS_PER_DAY * frac + ((C3 * c + C2) * c + C1) * c;

    Angle::from_radians(reduce_seconds_to_radians(seconds))
}

/// Arcseconds in one revolution.
const ARCSEC_PER_TURN: f64 = 1_296_000.0;

/// The kinematic terms of the IAU 1982 equation of the equinoxes,
/// `0.00264″ sin Ω + 0.000063″ sin 2Ω`.
///
/// `Ω` is the IAU 1980 mean longitude of the Moon's ascending node,
/// `125°02′40.280″ − (5ʳ + 134°08′10.539″)·T + 7.455″·T² + 0.008″·T³`, `T` in Julian centuries of
/// TT from J2000.
///
/// TEME needs these and nothing else of the equation of the equinoxes: Eq. (C-1) of Vallado et al.,
/// AIAA 2006-6753, rotates TEME into the pseudo-Earth-fixed frame by `GMST1982 + EqEquinox1982_kin`,
/// and the authors confirmed (2026-10-28) that (C-1), not the paper's Eq. (1), is the intended
/// transform. `derivations/teme_c1.mac` evaluates both at 40 digits on the paper's example: the
/// angle is 0.001 850″ there, and it moves the example's position by 7.1 cm.
#[must_use]
pub fn equinox_kinematic_1982(tt: Epoch<Tt>) -> Angle {
    let t = tt.julian_centuries();
    let node = ((0.008 * t + 7.455) * t - (5.0 * ARCSEC_PER_TURN + 482_890.539)) * t + 450_160.280;
    let node = Angle::from_arcseconds(libm::fmod(node, ARCSEC_PER_TURN)).radians();
    Angle::from_arcseconds(0.002_64 * libm::sin(node) + 0.000_063 * libm::sin(2.0 * node))
}

/// Reduce an angle expressed in seconds of time into `[0, 2π)` radians.
fn reduce_seconds_to_radians(seconds: f64) -> f64 {
    let mut reduced = seconds - SECONDS_PER_DAY * libm::floor(seconds / SECONDS_PER_DAY);

    // `floor` of a quotient can land on the wrong side of the boundary when the
    // quotient is within an ulp of an integer. Clamp rather than trust it.
    if reduced >= SECONDS_PER_DAY {
        reduced -= SECONDS_PER_DAY;
    } else if reduced < 0.0 {
        reduced += SECONDS_PER_DAY;
    }
    debug_assert!((0.0..SECONDS_PER_DAY).contains(&reduced));

    reduced * (Angle::TAU / SECONDS_PER_DAY)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // a panicking test is a failing test

    use super::*;
    use spacelab_core::time::NS_PER_DAY;

    /// One nanosecond of Earth rotation is 7.3 × 10⁻¹⁴ rad. A tolerance of
    /// 10⁻⁹ rad is 14 µs of time — far tighter than any TLE-derived UT1, and far
    /// looser than our arithmetic. It tests the formula, not the floating point.
    const TOL_RAD: f64 = 1e-9;

    fn at_days(days: f64) -> Epoch<Ut1> {
        Epoch::from_days_since_j2000(days)
    }

    /// The coefficients are the nearest `f64` to the published decimals — the exact
    /// bit patterns `derivations/gmst82.py` emits (50-digit mpmath, < 0.5 ulp each).
    /// A mistyped digit changes a bit pattern and fails here, not silently in orbit.
    ///
    /// (G3) `derivations/gmst82.mac` decodes the four patterns and certifies each within half an
    /// ulp of its published decimal in exact rationals; `derivations/gmst82.gp` repeats it.
    #[test]
    fn coefficients_are_the_certified_nearest_f64() {
        assert_eq!(C0.to_bits(), 0x40F0_6EE8_C649_906D);
        assert_eq!(C1.to_bits(), 0x4160_7AD7_1A02_FF8F);
        assert_eq!(C2.to_bits(), 0x3FB7_D5A9_EB20_74EB);
        assert_eq!(C3.to_bits(), 0xBEDA_0133_05E6_C9CE);
    }

    #[test]
    fn gmst_at_j2000_is_the_published_value() {
        // 280.46061837504° is the standard published GMST at J2000.0.
        let expected = 280.460_618_375_04_f64.to_radians();
        let got = gmst82(Epoch::<Ut1>::J2000).radians();
        assert!(
            (got - expected).abs() < TOL_RAD,
            "got {got}, expected {expected}, delta {}",
            got - expected
        );
    }

    #[test]
    fn gmst_always_lies_in_one_turn() {
        for d in [-40_000.0, -1.5, -0.25, 0.0, 0.5, 1.0, 9_000.0, 40_000.0] {
            let g = gmst82(at_days(d)).radians();
            assert!((0.0..Angle::TAU).contains(&g), "day {d} gave {g}");
        }
    }

    #[test]
    fn earth_turns_360_9856_degrees_per_solar_day() {
        // The whole-day term contributes nothing modulo a turn, so a one-day step
        // advances GMST by exactly the linear coefficient over a century:
        //   8640184.812866 / 36525 s = 236.555367...s = 0.9856473°
        let expected = 0.985_647_3_f64.to_radians();

        for start in [-3_650.0, 0.0, 1_234.5, 7_305.0] {
            let a = gmst82(at_days(start)).radians();
            let b = gmst82(at_days(start + 1.0)).radians();
            let advance = (b - a).rem_euclid(Angle::TAU);
            assert!(
                (advance - expected).abs() < 1e-6,
                "start {start}: advance {advance}, expected {expected}"
            );
        }
    }

    /// GMST is an angle. It increases monotonically *modulo a turn*, not in
    /// `[0, 2π)` — it passes through zero roughly once a day. Asserting plain
    /// monotonicity would be asserting that the Earth stops rotating at midnight.
    #[test]
    fn gmst_advances_at_the_sidereal_rate() {
        // One day of UT1 advances GMST by 86400 + 8640184.812866/36525 seconds of
        // angle. The ratio of the two is the sidereal rate:
        //   1 + 8640184.812866 / (36525 × 86400) = 1.00273790935...
        const SIDEREAL_RATE: f64 = 1.0 + 8_640_184.812_866 / (36_525.0 * 86_400.0);

        let step = NS_PER_DAY / 64;
        let expected = (Angle::TAU / 64.0) * SIDEREAL_RATE;

        let mut previous = gmst82(Epoch::<Ut1>::from_ns_since_j2000(0)).radians();
        for k in 1..200_i128 {
            let g = gmst82(Epoch::<Ut1>::from_ns_since_j2000(k * step)).radians();
            let advance = (g - previous).rem_euclid(Angle::TAU);
            assert!(
                (advance - expected).abs() < 1e-9,
                "step {k}: advance {advance}, expected {expected}"
            );
            previous = g;
        }
    }

    /// `floor` of the quotient can land a whole day past the argument, and the correction
    /// after it is what puts the residue back inside the day.
    ///
    /// A hair below zero is the case: `-1e-13 / 86400` is `-1.16e-18`, whose floor is `-1`,
    /// and `-1e-13 + 86400` rounds to exactly `86400` — one full turn, not none. The
    /// residue is nowhere near a boundary for any UT1 [`gmst82`] is given, so the guard is
    /// tested here rather than through the public function.
    #[test]
    fn the_reduction_puts_a_day_boundary_at_the_start_of_the_day_not_the_end() {
        for seconds in [-1e-13, -1e-14, -1e-16] {
            let a = reduce_seconds_to_radians(seconds);
            assert!(
                (0.0..1e-12).contains(&a),
                "{seconds:e} s reduced to {a}, not to the start of the day"
            );
        }
    }

    #[test]
    fn reduction_is_exact_at_the_boundaries() {
        assert!((reduce_seconds_to_radians(0.0) - 0.0).abs() < 1e-15);
        let just_under = reduce_seconds_to_radians(SECONDS_PER_DAY - 1e-6);
        assert!(just_under < Angle::TAU && just_under > Angle::TAU - 1e-9);
        assert!(reduce_seconds_to_radians(SECONDS_PER_DAY).abs() < 1e-12);
        assert!(reduce_seconds_to_radians(-1.0) > Angle::TAU - 1e-3);
    }

    /// (G2) GMST is the defining polynomial, reduced to a turn. `derivations/gmst82.mac` shows the
    /// coded split differs from the printed Eq. 3-47 by whole days only, and evaluates the coded form
    /// in exact rationals at these UT1 epochs (whole nanoseconds from J2000, so the epoch is exactly
    /// what the test passes); `derivations/gmst82.gp` evaluates the printed polynomial itself, whole-day
    /// term and all, and agrees to 5e-40 rad. The epochs span fifty centuries before J2000 to a
    /// century after it, through Vallado's Example 3-5 epoch, 1992 Aug 20 12:14 UT1.
    ///
    /// The bound is the conditioning the function's own documentation derives, a few `1.4e-13·|T|` rad
    /// from the linear term plus an ulp of the turn: measured 0 at J2000, 3.1e-15 and 4.9e-15 rad
    /// within a quarter century, 2.1e-13 rad at one century and 3.9e-12 rad at fifty — 1.5 and 0.56
    /// of `1.4e-13·|T|`, so four of them is the bound.
    #[test]
    #[allow(clippy::excessive_precision, clippy::unreadable_literal)]
    fn gmst_is_the_exact_rational_polynomial_reduced_to_a_turn() {
        let cases: [(i128, f64, f64); 5] = [
            (0, 0.0, 4.894961212823058751375704431175952857581),
            (
                -232_415_160_000_000_000,
                -7.364791999391588713970644155449083580501e-2,
                2.663002216713348436839310469824036728281,
            ),
            (
                777_664_800_000_000_000,
                2.464271047227926078028747433264887063655e-1,
                1.082560959309939194850505924209108106535,
            ),
            (
                3_155_760_012_000_000_000,
                1.000000003802570537683474028443227621872,
                4.909283007871094720178042192396826163741,
            ),
            (
                -157_788_025_920_000_000_000,
                -5.000000821355236139630390143737166324435e1,
                2.349829389055239016345653714577592776064,
            ),
        ];
        for (ns, centuries, want) in cases {
            let got = gmst82(Epoch::<Ut1>::from_ns_since_j2000(ns)).radians();
            let bound = 4.0 * 1.4e-13 * centuries.abs() + 1e-15;
            assert!(
                (got - want).abs() < bound,
                "T = {centuries}: {got} against {want}"
            );
        }
    }

    /// `derivations/teme_c1.mac` at 40 digits, at the AIAA 2006-6753 Appendix C epoch
    /// (2004-04-06 07:51:28.386 UTC, TT = UTC + 64.184 s): `Ω` = 42.604 613 979 722°, and the
    /// kinematic terms 0.001 849 888 944 514 253″. The tolerance is the epoch's own: a day count
    /// near 1557 in `f64` resolves 2e-11 days, which moves `Ω` by 1e-12 rad.
    #[test]
    fn kinematic_equinox_terms_match_the_cas_at_the_appendix_c_epoch() {
        let tt = Epoch::<Tt>::from_days_since_j2000(1556.5 + (28_288.386 + 64.184) / 86_400.0);
        let got = equinox_kinematic_1982(tt).arcseconds();
        let want = 1.849_888_944_514_253e-3;
        assert!((got - want).abs() < 1e-15, "{got:.18e} against {want:.18e}");
    }
}
