//! Complex arithmetic and a radix-2 Cooley–Tukey FFT — the spectral backbone of SAR focusing,
//! interferometry and tomography. `alloc` for scratch; power-of-two lengths only (callers zero-pad).

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
    #![allow(clippy::unreadable_literal, clippy::float_cmp)]
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
}
