//! Complex arithmetic and a radix-2 Cooley–Tukey FFT — the spectral backbone of SAR focusing,
//! interferometry and tomography. `alloc` for scratch; power-of-two lengths only (callers zero-pad).
//!
//! Gate: `derivations/fft_radix2.mac` shows on `N = 8`, in closed radicals, that the butterflies are
//! the Danielson–Lanczos split of the DFT `X_k = Σ x_n e^{−2πink/N}`, that `conj(F)·F = N·I` and
//! Parseval hold, and that the swap loop is the bit-reversal order; `derivations/fft_radix2.gp`
//! re-checks the split, the inverse and Parseval in exact `Q(ζ_N)` arithmetic for `N = 4 … 64` and
//! computes the reference transform. The module tests
//! `the_fft_is_the_dft_of_an_exact_gaussian_integer_input` (G2), `ifft_inverts_fft` (G1) and
//! `the_fft_keeps_parseval_s_energy` (G4) hold the shipped code to them.

// The butterfly's single-char names (a, u, v, w, i, j, k) are the standard FFT notation; `n as f64` is
// in range for any realistic transform size.
#![allow(clippy::many_single_char_names, clippy::cast_precision_loss)]

use alloc::vec::Vec;
use core::ops::{Add, Mul, Sub};

/// A complex number `re + i·im`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// The additive identity `0 + 0i`.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Construct from parts.
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// `r·e^{iθ}` — polar form.
    #[must_use]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self::new(r * libm::cos(theta), r * libm::sin(theta))
    }

    /// The complex conjugate `re − i·im`.
    #[must_use]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// The squared magnitude `re² + im²`.
    #[must_use]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// The magnitude `|z|`.
    #[must_use]
    pub fn abs(self) -> f64 {
        libm::sqrt(self.norm_sqr())
    }

    /// The argument `atan2(im, re)` (radians).
    #[must_use]
    pub fn arg(self) -> f64 {
        libm::atan2(self.im, self.re)
    }

    /// Scale by a real factor.
    #[must_use]
    pub fn scale(self, k: f64) -> Self {
        Self::new(self.re * k, self.im * k)
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}
impl Sub for Complex {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}
impl Mul for Complex {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

/// In-place forward FFT (radix-2 Cooley–Tukey). `data.len()` must be a power of two.
pub fn fft(data: &mut [Complex]) {
    transform(data, false);
}

/// In-place inverse FFT (unitary `1/N` scaling). `data.len()` must be a power of two.
pub fn ifft(data: &mut [Complex]) {
    transform(data, true);
    let inv_n = 1.0 / data.len() as f64;
    for x in data.iter_mut() {
        *x = x.scale(inv_n);
    }
}

/// Zero-pad `data` to `n` (`n ≥ data.len()`), returning a fresh buffer.
#[must_use]
pub fn zero_pad(data: &[Complex], n: usize) -> Vec<Complex> {
    let mut out = alloc::vec![Complex::ZERO; n];
    out[..data.len()].copy_from_slice(data);
    out
}

fn transform(a: &mut [Complex], inverse: bool) {
    let n = a.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of two");
    if n <= 1 {
        return;
    }
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    // Butterflies.
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let ang = sign * core::f64::consts::TAU / len as f64;
        let wlen = Complex::from_polar(1.0, ang);
        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = a[i + k];
                let v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w = w * wlen;
            }
            i += len;
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unreadable_literal,
        clippy::float_cmp,
        clippy::excessive_precision
    )]
    use super::*;

    fn close(a: Complex, b: Complex, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn fft_matches_numpy() {
        // np.fft.fft([1,2,3,4,5,6,7,8]).
        let mut x: Vec<Complex> = (1..=8).map(|k| Complex::new(f64::from(k), 0.0)).collect();
        let expect = [
            Complex::new(36.0, 0.0),
            Complex::new(-4.0, 9.65685424949238),
            Complex::new(-4.0, 4.0),
            Complex::new(-4.0, 1.6568542494923806),
            Complex::new(-4.0, 0.0),
            Complex::new(-4.0, -1.6568542494923806),
            Complex::new(-4.0, -4.0),
            Complex::new(-4.0, -9.65685424949238),
        ];
        fft(&mut x);
        for (a, b) in x.iter().zip(&expect) {
            assert!(close(*a, *b, 1e-10), "{a:?} vs {b:?}");
        }
    }

    /// (G1) `ifft` undoes `fft`: `conj(F)·F = N·I` (`derivations/fft_radix2.mac`, exact in
    /// `Q(ζ_N)` in `derivations/fft_radix2.gp`), so the `1/N` belongs on the inverse alone.
    #[test]
    fn ifft_inverts_fft() {
        let orig: Vec<Complex> = (0..16)
            .map(|k| {
                Complex::new(
                    libm::sin(0.3 * f64::from(k)),
                    libm::cos(0.17 * f64::from(k)),
                )
            })
            .collect();
        let mut x = orig.clone();
        fft(&mut x);
        ifft(&mut x);
        for (a, b) in x.iter().zip(&orig) {
            assert!(close(*a, *b, 1e-12), "round-trip {a:?} vs {b:?}");
        }
    }

    /// The input of the gate below: `x_n = ((n mod 5) − 2) + i((3n mod 7) − 3)`, Gaussian
    /// integers, so every reference value is an exact element of `Q(ζ₁₆)`.
    fn gaussian_integer_input() -> Vec<Complex> {
        (0..16)
            .map(|n: i32| Complex::new(f64::from(n % 5 - 2), f64::from((3 * n) % 7 - 3)))
            .collect()
    }

    /// (G2) The shipped radix-2 transform is the DFT `X_k = Σ x_n e^{−2πink/16}` of an exact input.
    ///
    /// The references are the definition summed in exact cyclotomic arithmetic by
    /// `derivations/fft_radix2.gp` and in 40-digit bigfloats by `derivations/fft_radix2.mac`, which
    /// agree to 1e-39. Sixteen points take the butterflies through four stages, so the twiddle
    /// recurrence `w ← w·w_len` runs up to `w⁷` — the numpy fixture above, at eight real points,
    /// never transforms a complex input.
    #[test]
    fn the_fft_is_the_dft_of_an_exact_gaussian_integer_input() {
        const DFT: [(f64, f64); 16] = [
            (-2.0, -3.0),
            (-0.3529701345641262719275, -2.986330253937075947743),
            (6.656854249492380195207, -2.828427124746190097603),
            (-14.65867108169942197248, 0.2483028813327445088006),
            (6.0, -5.0),
            (-4.906749931920417084326, -5.165910681040350540001),
            (-1.343145750507619804793, -8.828427124746190097603),
            (-14.25471763208691681834, -0.4005438163101709965442),
            (2.0, -3.0),
            (20.73774644541436190635, -5.599456183689829003456),
            (-4.656854249492380195207, 2.828427124746190097603),
            (4.901311768818707118884, -0.8340893189596494599989),
            (-2.0, -1.0),
            (-11.47802637892981855010, -6.248302881332744508801),
            (-12.65685424949238019521, -3.171572875253809902397),
            (-3.987923055032368328061, -3.013669746062924052257),
        ];
        let mut x = gaussian_integer_input();
        let scale: f64 = x.iter().map(|z| z.abs()).sum();
        fft(&mut x);
        let mut worst = 0.0_f64;
        for (k, (got, &(re, im))) in x.iter().zip(&DFT).enumerate() {
            let d = (*got - Complex::new(re, im)).abs();
            // log2(16) butterfly stages, each rounding relative to the input's L1 norm.
            assert!(
                d <= 16.0 * f64::EPSILON * scale,
                "X[{k}] = {got:?}, want ({re}, {im}): off by {d:e}"
            );
            worst = worst.max(d);
        }
        println!("fft vs exact DFT: worst {worst:.3e} against an L1 norm of {scale}");
    }

    /// (G4) Parseval, `Σ|X_k|² = N·Σ|x_n|²` (`derivations/fft_radix2.mac` for a symbolic input,
    /// exact in `Q(ζ_N)` up to `N = 64` in `derivations/fft_radix2.gp`), for the shipped transform.
    #[test]
    fn the_fft_keeps_parseval_s_energy() {
        for len in [2usize, 4, 16, 64, 256] {
            let x: Vec<Complex> = (0..len)
                .map(|n| {
                    let t = n as f64;
                    Complex::new(libm::sin(0.7 * t) + 0.25, libm::cos(1.3 * t * t))
                })
                .collect();
            let before: f64 = x.iter().map(|z| z.norm_sqr()).sum();
            let mut y = x.clone();
            fft(&mut y);
            let after: f64 = y.iter().map(|z| z.norm_sqr()).sum::<f64>() / len as f64;
            assert!(
                (after / before - 1.0).abs() < 1e-14,
                "N = {len}: sum |x|^2 = {before}, sum |X|^2 / N = {after}"
            );
        }
    }
}
