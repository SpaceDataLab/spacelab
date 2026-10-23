//! Reductions for the workspace's accuracy tests.
//!
//! Nothing here belongs in a shipped binary; the crate is `publish = false` and is only ever a
//! `dev-dependency`. It exists because the tests across fifteen crates share one comparison shape —
//! sweep a set of inputs, measure how far the implementation is from an oracle at each, reduce to
//! the worst, assert the worst is small — and that shape has a floating-point trap in it that a
//! per-file helper would have to be rediscovered and re-fixed in each file.
//!
//! # The trap
//!
//! `f64::max` is IEEE-754 `maxNum`: given one NaN operand it returns the *other* one. So
//!
//! ```ignore
//! let mut worst = 0.0;
//! for x in inputs { worst = worst.max(error_at(x)); }
//! assert!(worst < tolerance);
//! ```
//!
//! passes with `worst == 0.0` when the implementation returns NaN at *every* input. The reduction
//! discards precisely the sample the test most needs to see, and it discards it silently: a
//! transform that has stopped producing numbers at all scores a perfect zero error. This was
//! measured, not reasoned about — with the TT epoch of `spacelab-frames`'s CIO interpolator
//! corrupted, all six of its out-of-grid queries returned NaN and the assertion still passed.
//!
//! [`nan_max`] is the same reduction with the trap removed.

#![no_std]

/// The larger of `a` and `b`, keeping a NaN instead of discarding it.
///
/// Use this, not `f64::max`, to reduce a sequence of measured errors to the worst one. Once a NaN
/// enters the running value it stays, so the assertion that consumes it fails — `NaN < tolerance`
/// is false — and names the comparison that produced it rather than reporting perfect agreement.
///
/// The sign of the NaN is not consulted: `f64::max` would keep a positive NaN and drop a negative
/// one, which is a distinction no accuracy test means to draw.
#[must_use]
pub fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// How many representable doubles lie between `got` and `want`.
///
/// Zero means the two are the same double. The count is exact across zero and across the
/// subnormals, and `-0.0` and `+0.0` are the same point; NaN on either side gives [`i64::MAX`],
/// which no tolerance accepts.
///
/// This is the measure to use when a computation is expected to land on the correctly rounded
/// answer, or within a fixed number of rounding steps of it. A relative tolerance cannot say that:
/// near a root, near a cancellation, or anywhere the result is small compared with its inputs,
/// "1e-15 relative" is thousands of ulps in one place and unreachable in another.
#[must_use]
pub fn ulps_between(got: f64, want: f64) -> i64 {
    // Map the doubles onto a monotone integer key — sign-magnitude reordered into two's complement
    // — so that the difference of two keys counts the doubles between them whatever their signs.
    fn key(x: f64) -> i64 {
        #[expect(clippy::cast_possible_wrap, reason = "the wrap is the reordering")]
        let bits = x.to_bits() as i64;
        if bits < 0 {
            i64::MIN.wrapping_sub(bits)
        } else {
            bits
        }
    }
    if got.is_nan() || want.is_nan() {
        return i64::MAX;
    }
    let d = (i128::from(key(got)) - i128::from(key(want))).abs();
    i64::try_from(d).unwrap_or(i64::MAX)
}

/// FNV-1a over the IEEE-754 bits of a sequence of doubles — a seal for a machine-generated
/// coefficient table.
///
/// A table transcribed from a published model is verified by its provenance, not by anything an
/// accuracy test can assert. Flip the sign of one degree-85 gravity coefficient and no oracle in
/// this workspace resolves the difference: the term is orders of magnitude below the agreement the
/// field tests already demand, so every one of them still passes. What a test *can* see is that the
/// bytes are no longer the ones the generator wrote. That is what this seals, and it is the only
/// assertion that makes "generated — do not hand-edit" enforceable rather than advisory.
///
/// The bits, not the values, so that a `-0.0` written where `0.0` belongs is a difference; and the
/// order matters, so a permuted table breaks the seal as surely as an edited one.
#[must_use]
pub fn fnv1a_f64(values: impl IntoIterator<Item = f64>) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Assert that `got` is at most `tolerance` representable doubles from `want`.
///
/// `what` names the quantity in the failure message; the message carries both values at full
/// precision and the measured distance, so a failure says how far off it is rather than only that
/// it is off.
///
/// # Panics
///
/// If the two are further apart than `tolerance`, or if either is NaN.
#[track_caller]
pub fn assert_within_ulps(got: f64, want: f64, tolerance: i64, what: &str) {
    let d = ulps_between(got, want);
    assert!(
        d <= tolerance,
        "{what}: got {got:.17e}, want {want:.17e}, {d} ulps apart (allowed {tolerance})"
    );
}

#[cfg(test)]
mod tests {
    // `f64::max(3.0, NAN)` is exactly 3.0 by IEEE-754 `maxNum`; that exact equality is the trap
    // being demonstrated, so comparing it is the point rather than an accident of precision.
    #![allow(clippy::float_cmp)]

    use super::{assert_within_ulps, fnv1a_f64, nan_max, ulps_between};

    /// The assertion helper itself — the one thing in this crate whose whole contract is that it
    /// panics. Every other test here checks `ulps_between`, so deleting the body of the helper that
    /// wraps it changed nothing observable, in this crate or in the crates that budget ulps with it.
    #[test]
    fn the_ulp_assertion_accepts_what_is_inside_the_tolerance() {
        assert_within_ulps(1.0, 1.0, 0, "identical");
        assert_within_ulps(1.0, f64::from_bits(1.0f64.to_bits() + 2), 2, "two ulps");
    }

    #[test]
    #[should_panic(expected = "3 ulps apart")]
    fn the_ulp_assertion_fires_beyond_the_tolerance() {
        assert_within_ulps(1.0, f64::from_bits(1.0f64.to_bits() + 3), 2, "three ulps");
    }

    #[test]
    #[should_panic(expected = "a nan")]
    fn a_nan_is_beyond_every_tolerance() {
        assert_within_ulps(f64::NAN, 1.0, i64::MAX - 1, "a nan");
    }

    #[test]
    fn the_ulp_distance_counts_doubles_and_spans_zero_and_the_signs() {
        assert_eq!(ulps_between(1.0, 1.0), 0);
        assert_eq!(ulps_between(0.0, -0.0), 0, "the two zeros are one point");
        assert_eq!(ulps_between(1.0, f64::from_bits(1.0f64.to_bits() + 1)), 1);
        // Across zero: the subnormals either side of it are one apart from it and two from each other.
        assert_eq!(ulps_between(f64::MIN_POSITIVE, 0.0), 1 << 52);
        assert_eq!(ulps_between(5e-324, -5e-324), 2);
        // Symmetric, and the same count in either direction.
        assert_eq!(ulps_between(-1.0, -1.5), ulps_between(1.5, 1.0));
        // Far apart is a number, not an overflow; NaN is beyond every tolerance.
        assert!(ulps_between(f64::MIN, f64::MAX) > 0);
        assert_eq!(ulps_between(f64::NAN, 1.0), i64::MAX);
        assert_eq!(ulps_between(1.0, f64::NAN), i64::MAX);
    }

    #[test]
    fn a_nan_survives_the_reduction_from_either_side_and_with_either_sign() {
        // `f64::max` drops all four of these; that is the whole reason the function exists.
        assert!(nan_max(3.0, f64::NAN).is_nan());
        assert!(nan_max(f64::NAN, 3.0).is_nan());
        assert!(nan_max(3.0, -f64::NAN).is_nan());
        assert!(nan_max(-f64::NAN, 3.0).is_nan());
        assert!(f64::max(3.0, f64::NAN) == 3.0, "the trap being avoided");
    }

    #[test]
    fn a_nan_that_has_entered_the_running_value_is_never_washed_out() {
        // The failure mode a one-shot check misses: NaN at sample 1, finite everywhere after.
        let mut worst = 0.0_f64;
        for sample in [f64::NAN, 1.0, 2.0, 3.0] {
            worst = nan_max(worst, sample);
        }
        assert!(worst.is_nan(), "a NaN sample must reach the assertion");
        assert!(
            worst.partial_cmp(&10.0).is_none(),
            "and must be incomparable to a tolerance, so `worst < tol` is false"
        );
    }

    #[test]
    fn the_seal_is_fnv_1a_and_separates_the_ways_a_table_can_be_corrupted() {
        // The four references come from an independent evaluation of FNV-1a over `struct.pack('<d')`
        // in Python, not from this implementation: a seal checked against its own output seals
        // nothing. The empty fold is the published 64-bit offset basis.
        assert_eq!(fnv1a_f64([]), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_f64([1.0, -2.5, 0.0]), 0xafda_801c_a49b_e31c);
        // A flipped sign -- the mutation a generated table is exposed to -- and a permutation and a
        // signed zero all have to move the seal, or it would pass over a corrupted table.
        assert_eq!(fnv1a_f64([1.0, 2.5, 0.0]), 0xfaaf_d152_5720_d99c);
        assert_eq!(fnv1a_f64([0.0, -2.5, 1.0]), 0x4691_ed63_cdde_21dc);
        assert_eq!(fnv1a_f64([1.0, -2.5, -0.0]), 0xafdb_001c_a49c_bc9c);
    }

    #[test]
    fn without_a_nan_it_is_exactly_f64_max() {
        for a in [-1e300, -1.0, -0.0, 0.0, 1.0, 1e300, f64::INFINITY] {
            for b in [-1e300, -1.0, -0.0, 0.0, 1.0, 1e300, f64::NEG_INFINITY] {
                let (got, want) = (nan_max(a, b), a.max(b));
                assert!(got.to_bits() == want.to_bits(), "nan_max({a}, {b}) = {got}");
            }
        }
    }
}
