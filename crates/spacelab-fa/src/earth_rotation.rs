//! Rotation of the Earth about its axis.

use spacelab_core::{
    Angle,
    time::{Epoch, SECONDS_PER_DAY, Ut1},
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
}
