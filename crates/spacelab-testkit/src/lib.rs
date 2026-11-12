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

/// SHA-256 (FIPS 180-4) of `bytes`, as the 64 lowercase hexadecimal digits `sha256sum` prints — a seal
/// for a committed excerpt of a third-party file.
///
/// A test that parses an excerpt proves the parser on those bytes; the seal proves they are still the
/// bytes that were cut from the provider's file, and anyone can check the pinned value with the
/// standard tool, without this workspace.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> [u8; 64] {
    // §5.3.3: the first 32 bits of the fractional parts of the square roots of the first eight primes.
    let mut state: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let (blocks, rest) = bytes.as_chunks::<64>();
    for block in blocks {
        sha256_block(&mut state, block);
    }
    // §5.1.1: a one bit, zeros to 56 bytes modulo 64, then the message length in bits, big-endian.
    let mut tail = [0_u8; 128];
    tail[..rest.len()].copy_from_slice(rest);
    tail[rest.len()] = 0x80;
    let end = if rest.len() < 56 { 64 } else { 128 };
    let bits = (bytes.len() as u64).wrapping_mul(8);
    tail[end - 8..end].copy_from_slice(&bits.to_be_bytes());
    for block in tail[..end].as_chunks::<64>().0 {
        sha256_block(&mut state, block);
    }
    let digits = b"0123456789abcdef";
    let mut out = [0_u8; 64];
    for (i, byte) in state.iter().flat_map(|w| w.to_be_bytes()).enumerate() {
        out[2 * i] = digits[usize::from(byte >> 4)];
        out[2 * i + 1] = digits[usize::from(byte & 0x0f)];
    }
    out
}

/// §4.2.2: the first 32 bits of the fractional parts of the cube roots of the first 64 primes,
/// generated in PARI/GP at 200 digits.
#[rustfmt::skip]
const SHA256_K: [u32; 64] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4, 0xab1c_5ed5,
    0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174,
    0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
    0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7, 0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967,
    0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85,
    0xa2bf_e8a1, 0xa81a_664b, 0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
    0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
];

/// One 64-byte block of §6.2.2's computation.
#[expect(
    clippy::many_single_char_names,
    reason = "FIPS 180-4's own names for the working variables"
)]
fn sha256_block(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0_u32; 64];
    for (word, bytes) in w.iter_mut().zip(block.as_chunks::<4>().0) {
        *word = u32::from_be_bytes(*bytes);
    }
    for t in 16..64 {
        let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
        let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
        w[t] = w[t - 16]
            .wrapping_add(s0)
            .wrapping_add(w[t - 7])
            .wrapping_add(s1);
    }
    let mut v = *state;
    for (k, wt) in SHA256_K.iter().zip(w) {
        let [a, b, c, d, e, f, g, h] = v;
        let t1 = h
            .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
            .wrapping_add((e & f) ^ (!e & g))
            .wrapping_add(*k)
            .wrapping_add(wt);
        let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
            .wrapping_add((a & b) ^ (a & c) ^ (b & c));
        v = [t1.wrapping_add(t2), a, b, c, d.wrapping_add(t1), e, f, g];
    }
    for (s, x) in state.iter_mut().zip(v) {
        *s = s.wrapping_add(x);
    }
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

    use super::{assert_within_ulps, fnv1a_f64, nan_max, sha256_hex, ulps_between};

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

    /// FIPS 180-2's three examples (one block, two blocks, a million bytes), the empty message, and
    /// lengths either side of the padding's one- and two-block boundaries, against `sha256sum`.
    #[test]
    fn sha256_is_the_standard_digest_at_every_padding_boundary() {
        static A: [u8; 1_000_000] = [b'a'; 1_000_000];
        let a = &A;
        for (message, digest) in [
            (
                &b"abc"[..],
                b"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                b"",
                b"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                b"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
            (
                &a[..],
                b"cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
            ),
            (
                &a[..55],
                b"9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
            ),
            (
                &a[..56],
                b"b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
            ),
            (
                &a[..63],
                b"7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34",
            ),
            (
                &a[..64],
                b"ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
            ),
            (
                &a[..119],
                b"31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb",
            ),
            (
                &a[..120],
                b"2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c",
            ),
        ] {
            assert_eq!(&sha256_hex(message), digest, "{} bytes", message.len());
        }
    }
}
