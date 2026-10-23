//! The `Scalar` trait: write the geometry once, evaluate it in `f64` or in [`Dual`].

use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::{Dual, HyperDual};

/// A real scalar that carries whatever derivative information it holds through arithmetic and the
/// transcendental functions a force model or measurement geometry needs.
///
/// Implemented for `f64` (no derivatives, the plain value) and for [`Dual<N>`] (value plus the `N`
/// partials). A function generic over `Scalar` is one source that yields the value when instantiated
/// at `f64` and the Jacobian when instantiated at `Dual<N>` — they cannot drift apart because they
/// are the same code.
pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Lift a constant into the scalar (zero derivatives for a dual).
    #[must_use]
    fn from_f64(x: f64) -> Self;
    /// The underlying value.
    #[must_use]
    fn value(self) -> f64;
    /// Square root.
    #[must_use]
    fn sqrt(self) -> Self;
    /// Reciprocal.
    #[must_use]
    fn recip(self) -> Self;
    /// Real power.
    #[must_use]
    fn powf(self, p: f64) -> Self;
    /// Sine.
    #[must_use]
    fn sin(self) -> Self;
    /// Cosine.
    #[must_use]
    fn cos(self) -> Self;
    /// Exponential — the atmosphere-density and radiation-shadow models need it.
    #[must_use]
    fn exp(self) -> Self;
    /// Natural logarithm — the Shapiro delay's closed form is one.
    #[must_use]
    fn ln(self) -> Self;
    /// Two-argument arctangent `atan2(self, x)` — the angle measurements (right ascension,
    /// declination, azimuth, elevation) are built from it.
    #[must_use]
    fn atan2(self, x: Self) -> Self;
}

impl Scalar for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
    fn value(self) -> f64 {
        self
    }
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    fn recip(self) -> Self {
        1.0 / self
    }
    fn powf(self, p: f64) -> Self {
        libm::pow(self, p)
    }
    fn sin(self) -> Self {
        libm::sin(self)
    }
    fn cos(self) -> Self {
        libm::cos(self)
    }
    fn exp(self) -> Self {
        libm::exp(self)
    }
    fn ln(self) -> Self {
        libm::log(self)
    }
    fn atan2(self, x: Self) -> Self {
        libm::atan2(self, x)
    }
}

impl<const N: usize> Scalar for Dual<N> {
    fn from_f64(x: f64) -> Self {
        Dual::constant(x)
    }
    fn value(self) -> f64 {
        Dual::value(self)
    }
    fn sqrt(self) -> Self {
        Dual::sqrt(self)
    }
    fn recip(self) -> Self {
        Dual::recip(self)
    }
    fn powf(self, p: f64) -> Self {
        Dual::powf(self, p)
    }
    fn sin(self) -> Self {
        Dual::sin(self)
    }
    fn cos(self) -> Self {
        Dual::cos(self)
    }
    fn exp(self) -> Self {
        Dual::exp(self)
    }
    fn ln(self) -> Self {
        Dual::ln(self)
    }
    fn atan2(self, x: Self) -> Self {
        Dual::atan2(self, x)
    }
}

impl<const N: usize> Scalar for HyperDual<N> {
    fn from_f64(x: f64) -> Self {
        HyperDual::constant(x)
    }
    fn value(self) -> f64 {
        HyperDual::value(self)
    }
    fn sqrt(self) -> Self {
        HyperDual::sqrt(self)
    }
    fn recip(self) -> Self {
        HyperDual::recip(self)
    }
    fn powf(self, p: f64) -> Self {
        HyperDual::powf(self, p)
    }
    fn sin(self) -> Self {
        HyperDual::sin(self)
    }
    fn cos(self) -> Self {
        HyperDual::cos(self)
    }
    fn exp(self) -> Self {
        HyperDual::exp(self)
    }
    fn ln(self) -> Self {
        HyperDual::ln(self)
    }
    fn atan2(self, x: Self) -> Self {
        HyperDual::atan2(self, x)
    }
}
