//! The `f64` methods that code outside the `std`-gated modules calls and `core` does not provide.
//!
//! With `std` each is the inherent method, so a `std` build compiles to the code it did before;
//! without it each is its `libm` counterpart. The two compute the same IEEE 754 operation: `floor`
//! and `trunc` are exact, and `fma` is fusedMultiplyAdd, `a·b + c` rounded once (IEEE 754-2019
//! §5.4.1). So both builds return the same bits for every input whose result is not NaN. The tests
//! below hold each pair to that on random bit patterns and on the inputs where a wrong route would
//! show; `just no-std` holds the public kernels that call `mul_add` and `floor` to it.

/// `a·b + c`, rounded once.
#[inline]
pub(crate) fn mul_add(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::fma(a, b, c)
    }
}

/// The largest integer not above `x`.
#[inline]
pub(crate) fn floor(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.floor()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::floor(x)
    }
}

/// `x` less its integer part, `x − trunc(x)`: exact, and of the sign of `x`.
#[inline]
pub(crate) fn fract(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.fract()
    }
    #[cfg(not(feature = "std"))]
    {
        x - libm::trunc(x)
    }
}

#[cfg(test)]
mod tests {
    // The `std` build's routes, held to the `libm` routes a `no_std` build takes. A NaN result is held
    // only to be NaN: IEEE 754 leaves the payload of a propagated NaN to the implementation (§6.2.3).

    fn same(got: f64, want: f64) -> bool {
        got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())
    }

    /// `SplitMix64`: a fixed, seedable stream, so a failure names an input that reproduces.
    fn stream(mut state: u64) -> impl FnMut() -> u64 {
        move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    const SPECIAL: [f64; 12] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MIN_POSITIVE,
        5e-324,
        -5e-324,
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];

    /// `fma` against `mul_add` over random bit patterns, over operands of one scale, and over the
    /// cancellation `c ≈ −a·b`, where one rounding and two differ in every bit that survives. The
    /// sweep's own power is asserted: an unfused `a·b + c` must disagree with it on most of the
    /// cancelling triples, or it could not tell a fused route from an unfused one.
    #[test]
    fn mul_add_returns_the_bits_of_libm_fma() {
        // [0, 1) on 52 random bits: a mantissa under the exponent of 1.0, less 1.
        let mut next = stream(0x5EED_F32E);
        let mut unit = || f64::from_bits(0x3FF0_0000_0000_0000 | (next() >> 12)) - 1.0;
        let mut triples = Vec::new();
        for &a in &SPECIAL {
            for &b in &SPECIAL {
                for &c in &SPECIAL {
                    triples.push((a, b, c));
                }
            }
        }
        let mut bits = stream(0xB175);
        for _ in 0..100_000 {
            triples.push((
                f64::from_bits(bits()),
                f64::from_bits(bits()),
                f64::from_bits(bits()),
            ));
        }
        for _ in 0..100_000 {
            let a = (unit() - 0.5) * libm::exp2(libm::floor(unit() * 80.0) - 40.0);
            let b = (unit() - 0.5) * libm::exp2(libm::floor(unit() * 80.0) - 40.0);
            triples.push((a, b, (unit() - 0.5) * a * b * 4.0));
        }
        let mut cancelling = 0;
        let mut unfused_differs = 0;
        for k in 0..100_000 {
            let a = 1.0 + unit();
            let b = 1.0 + unit();
            let c = -(a * b) + f64::from(k % 5 - 2) * f64::EPSILON;
            cancelling += 1;
            if !same(a * b + c, libm::fma(a, b, c)) {
                unfused_differs += 1;
            }
            triples.push((a, b, c));
        }
        for &(a, b, c) in &triples {
            let (got, want) = (super::mul_add(a, b, c), libm::fma(a, b, c));
            assert!(
                same(got, want),
                "mul_add({a:e}, {b:e}, {c:e}) = {got:e} ({:#x}), libm::fma = {want:e} ({:#x})",
                got.to_bits(),
                want.to_bits()
            );
        }
        assert!(
            unfused_differs * 10 > cancelling * 9,
            "an unfused a·b + c differs on only {unfused_differs} of {cancelling} cancelling triples"
        );
    }

    /// `floor` and `fract` against `libm::floor` and `x − libm::trunc(x)` over random bit patterns
    /// and over values within an ulp of an integer, where a rounding route would show.
    #[test]
    fn floor_and_fract_return_the_bits_of_libm() {
        let mut bits = stream(0xF100);
        let mut xs: Vec<f64> = SPECIAL.to_vec();
        for _ in 0..200_000 {
            xs.push(f64::from_bits(bits()));
        }
        for n in -2_000..2_000 {
            let whole = f64::from(n) * 0.75;
            xs.extend([whole, whole.next_up(), whole.next_down()]);
        }
        for x in [
            4_503_599_627_370_495.5_f64,
            4_503_599_627_370_496.0,
            9e15,
            -9e15,
        ] {
            xs.extend([x, x.next_up(), x.next_down()]);
        }
        for &x in &xs {
            let (got, want) = (super::floor(x), libm::floor(x));
            assert!(
                same(got, want),
                "floor({x:e}) = {got:e}, libm::floor = {want:e}"
            );
            let (got, want) = (super::fract(x), x - libm::trunc(x));
            assert!(
                same(got, want),
                "fract({x:e}) = {got:e}, x - libm::trunc = {want:e}"
            );
        }
    }
}
