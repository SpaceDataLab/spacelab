//! The dual number: a value and its `N` partial derivatives.

#![allow(clippy::needless_range_loop)] // the loops index several arrays by one i; iterators would obscure them

use core::ops::{Add, Div, Mul, Neg, Sub};

/// A value carrying its `N` first partial derivatives, propagated by the exact chain rule.
///
/// `Copy` and stack-allocated: a Jacobian of a 6-state force model costs `Dual<6>` (56 bytes) per
/// intermediate, no allocation, and the derivative is exact to the same rounding as the value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual<const N: usize> {
    value: f64,
    grad: [f64; N],
}

impl<const N: usize> Dual<N> {
    /// A constant: value `v`, zero derivatives.
    #[must_use]
    pub const fn constant(v: f64) -> Self {
        Self {
            value: v,
            grad: [0.0; N],
        }
    }

    /// The `i`-th independent variable at value `v`: derivative 1 along axis `i`, 0 elsewhere.
    ///
    /// # Panics
    ///
    /// If `i >= N`.
    #[must_use]
    pub fn variable(v: f64, i: usize) -> Self {
        assert!(i < N, "variable index {i} out of range for Dual<{N}>");
        let mut grad = [0.0; N];
        grad[i] = 1.0;
        Self { value: v, grad }
    }

    /// Construct from a value and an explicit gradient.
    #[must_use]
    pub const fn new(value: f64, grad: [f64; N]) -> Self {
        Self { value, grad }
    }

    /// The value.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }

    /// The gradient (the `N` partial derivatives).
    #[must_use]
    pub const fn grad(self) -> [f64; N] {
        self.grad
    }

    /// Apply a differentiable unary function given `f(x)` and `f'(x)`: the chain rule in one place.
    #[inline]
    fn chain(self, f: f64, df: f64) -> Self {
        let mut grad = self.grad;
        for g in &mut grad {
            *g *= df;
        }
        Self { value: f, grad }
    }

    /// √. Derivative `1/(2√x)`.
    #[must_use]
    pub fn sqrt(self) -> Self {
        let r = libm::sqrt(self.value);
        self.chain(r, 0.5 / r)
    }

    /// Reciprocal `1/x`. Derivative `-1/x²`.
    #[must_use]
    pub fn recip(self) -> Self {
        let r = 1.0 / self.value;
        self.chain(r, -r * r)
    }

    /// Integer power. Derivative `n·xⁿ⁻¹`.
    #[must_use]
    pub fn powi(self, n: i32) -> Self {
        let f = libm::pow(self.value, f64::from(n));
        let df = f64::from(n) * libm::pow(self.value, f64::from(n - 1));
        self.chain(f, df)
    }

    /// Real power `xᵖ`. Derivative `p·xᵖ⁻¹`, formed as `p·f/x` so it is exactly consistent with the
    /// value and costs one `pow`, not two (`x > 0` at every call site).
    #[must_use]
    pub fn powf(self, p: f64) -> Self {
        let f = libm::pow(self.value, p);
        let df = p * f / self.value;
        self.chain(f, df)
    }

    /// Exponential. Derivative `eˣ`.
    #[must_use]
    pub fn exp(self) -> Self {
        let f = libm::exp(self.value);
        self.chain(f, f)
    }

    /// Natural logarithm. Derivative `1/x`.
    #[must_use]
    pub fn ln(self) -> Self {
        self.chain(libm::log(self.value), 1.0 / self.value)
    }

    /// Sine. Derivative `cos x`.
    #[must_use]
    pub fn sin(self) -> Self {
        self.chain(libm::sin(self.value), libm::cos(self.value))
    }

    /// Cosine. Derivative `-sin x`.
    #[must_use]
    pub fn cos(self) -> Self {
        self.chain(libm::cos(self.value), -libm::sin(self.value))
    }

    /// Absolute value. Derivative `sign(x)`; the kink at 0 takes derivative 0.
    #[must_use]
    pub fn abs(self) -> Self {
        let s = if self.value > 0.0 {
            1.0
        } else if self.value < 0.0 {
            -1.0
        } else {
            0.0
        };
        self.chain(libm::fabs(self.value), s)
    }

    /// Two-argument arctangent `atan2(y, x)`, with the correct partials
    /// `∂/∂y = x/(x²+y²)`, `∂/∂x = -y/(x²+y²)`.
    #[must_use]
    pub fn atan2(self, x: Self) -> Self {
        let y = self;
        let denom = x.value * x.value + y.value * y.value;
        let f = libm::atan2(y.value, x.value);
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = (x.value * y.grad[i] - y.value * x.grad[i]) / denom;
        }
        Self { value: f, grad }
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut grad = self.grad;
        for i in 0..N {
            grad[i] += rhs.grad[i];
        }
        Self {
            value: self.value + rhs.value,
            grad,
        }
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut grad = self.grad;
        for i in 0..N {
            grad[i] -= rhs.grad[i];
        }
        Self {
            value: self.value - rhs.value,
            grad,
        }
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    /// Product rule `(uv)' = u'v + uv'`.
    fn mul(self, rhs: Self) -> Self {
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = self.grad[i] * rhs.value + self.value * rhs.grad[i];
        }
        Self {
            value: self.value * rhs.value,
            grad,
        }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    /// Quotient rule `(u/v)' = (u'v - uv')/v²`.
    fn div(self, rhs: Self) -> Self {
        let inv = 1.0 / rhs.value;
        let inv2 = inv * inv;
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = (self.grad[i] * rhs.value - self.value * rhs.grad[i]) * inv2;
        }
        Self {
            value: self.value * inv,
            grad,
        }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut grad = self.grad;
        for g in &mut grad {
            *g = -*g;
        }
        Self {
            value: -self.value,
            grad,
        }
    }
}

// Scalar (f64) on the right, for ergonomic mixing with constants.
impl<const N: usize> Add<f64> for Dual<N> {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        Self {
            value: self.value + rhs,
            grad: self.grad,
        }
    }
}

impl<const N: usize> Sub<f64> for Dual<N> {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self {
        Self {
            value: self.value - rhs,
            grad: self.grad,
        }
    }
}

impl<const N: usize> Mul<f64> for Dual<N> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        let mut grad = self.grad;
        for g in &mut grad {
            *g *= rhs;
        }
        Self {
            value: self.value * rhs,
            grad,
        }
    }
}

impl<const N: usize> Div<f64> for Dual<N> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        self * (1.0 / rhs)
    }
}
