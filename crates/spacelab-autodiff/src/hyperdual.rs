//! The hyper-dual number: a value, its `N` first partials, and its `N×N` second partials.
//!
//! Where a [`Dual<N>`](crate::Dual) carries the gradient, a `HyperDual<N>` carries the gradient *and*
//! the Hessian, propagating both through every operation by the exact second-order chain rule. Seed
//! each input as a unit direction and one forward pass yields the value, the full Jacobian, and the
//! full Hessian of every output — the ingredients the second-order state-transition tensor needs,
//! exact to rounding rather than capped at the finite-difference floor (`√ε ≈ 1e-8`) a differenced
//! Jacobian would hit.
//!
//! Second partials of a `C²` function are symmetric, and every rule here preserves that by
//! construction: `hess[i][j]` and `hess[j][i]` are formed from the same commuted products, so the
//! stored `N×N` block is symmetric to the bit. The type is `Copy` and stack-allocated — a
//! `HyperDual<6>` is 344 bytes, and the force model is evaluated once per integration stage, so the
//! second-order pass is a small constant, not a hot inner loop.

#![allow(clippy::needless_range_loop)] // the loops index several arrays by one i/j; iterators would obscure them

use core::ops::{Add, Div, Mul, Neg, Sub};

/// A value carrying its `N` first partials and its `N×N` second partials, propagated by the exact
/// second-order chain rule.
///
/// The Hessian block is symmetric by construction (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperDual<const N: usize> {
    value: f64,
    grad: [f64; N],
    hess: [[f64; N]; N],
}

impl<const N: usize> HyperDual<N> {
    /// A constant: value `v`, zero first and second partials.
    #[must_use]
    pub const fn constant(v: f64) -> Self {
        Self {
            value: v,
            grad: [0.0; N],
            hess: [[0.0; N]; N],
        }
    }

    /// The `i`-th independent variable at value `v`: first derivative 1 along axis `i`, 0 elsewhere,
    /// all second derivatives 0.
    ///
    /// # Panics
    ///
    /// If `i >= N`.
    #[must_use]
    pub fn variable(v: f64, i: usize) -> Self {
        assert!(i < N, "variable index {i} out of range for HyperDual<{N}>");
        let mut grad = [0.0; N];
        grad[i] = 1.0;
        Self {
            value: v,
            grad,
            hess: [[0.0; N]; N],
        }
    }

    /// The value.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }

    /// The gradient (the `N` first partials).
    #[must_use]
    pub const fn grad(self) -> [f64; N] {
        self.grad
    }

    /// The Hessian (the `N×N` second partials), symmetric by construction.
    #[must_use]
    pub const fn hess(self) -> [[f64; N]; N] {
        self.hess
    }

    /// Apply a differentiable unary function from its value and first two derivatives at `self.value`:
    /// the second-order chain rule in one place. For `w = g(u)` with `f = g(u₀)`, `d1 = g'(u₀)`,
    /// `d2 = g''(u₀)`, the rules are `w' = d1·u'` and `w'' = d1·u'' + d2·u'⊗u'`.
    #[inline]
    fn chain(self, f: f64, d1: f64, d2: f64) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = d1 * self.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = d1 * self.hess[i][j] + d2 * self.grad[i] * self.grad[j];
            }
        }
        Self {
            value: f,
            grad,
            hess,
        }
    }

    /// √. `g' = 1/(2√x)`, `g'' = -1/(4 x^{3/2})`.
    #[must_use]
    pub fn sqrt(self) -> Self {
        let r = libm::sqrt(self.value);
        let d1 = 0.5 / r;
        let d2 = -0.25 / (self.value * r); // -¼ x^{-3/2} = -¼/(x·√x)
        self.chain(r, d1, d2)
    }

    /// Reciprocal `1/x`. `g' = -1/x²`, `g'' = 2/x³`.
    #[must_use]
    pub fn recip(self) -> Self {
        let r = 1.0 / self.value;
        self.chain(r, -r * r, 2.0 * r * r * r)
    }

    /// Real power `xᵖ` (`x > 0` at every call site). `g' = p·xᵖ⁻¹`, `g'' = p(p−1)·xᵖ⁻²`, formed from
    /// `f = xᵖ` as `p·f/x` and `p(p−1)·f/x²` so they cost one `pow`, not three.
    #[must_use]
    pub fn powf(self, p: f64) -> Self {
        let f = libm::pow(self.value, p);
        let d1 = p * f / self.value;
        let d2 = p * (p - 1.0) * f / (self.value * self.value);
        self.chain(f, d1, d2)
    }

    /// Integer power `xⁿ`. `g' = n·xⁿ⁻¹`, `g'' = n(n−1)·xⁿ⁻²`.
    #[must_use]
    pub fn powi(self, n: i32) -> Self {
        let f = libm::pow(self.value, f64::from(n));
        let d1 = f64::from(n) * libm::pow(self.value, f64::from(n - 1));
        let d2 = f64::from(n) * f64::from(n - 1) * libm::pow(self.value, f64::from(n - 2));
        self.chain(f, d1, d2)
    }

    /// Exponential. `g' = g'' = eˣ`.
    #[must_use]
    pub fn exp(self) -> Self {
        let f = libm::exp(self.value);
        self.chain(f, f, f)
    }

    /// Natural logarithm. `g' = 1/x`, `g'' = -1/x²`.
    #[must_use]
    pub fn ln(self) -> Self {
        let d1 = 1.0 / self.value;
        self.chain(libm::log(self.value), d1, -d1 * d1)
    }

    /// Sine. `g' = cos x`, `g'' = -sin x`.
    #[must_use]
    pub fn sin(self) -> Self {
        let (s, c) = (libm::sin(self.value), libm::cos(self.value));
        self.chain(s, c, -s)
    }

    /// Cosine. `g' = -sin x`, `g'' = -cos x`.
    #[must_use]
    pub fn cos(self) -> Self {
        let (s, c) = (libm::sin(self.value), libm::cos(self.value));
        self.chain(c, -s, -c)
    }

    /// Two-argument arctangent `atan2(y, x)` with `y = self`, carrying both partials to second order.
    ///
    /// The first partials are `∂/∂y = x/D`, `∂/∂x = -y/D` with `D = x²+y²`; the second-order term
    /// mixes the two inner arguments through `∂²/∂y²`, `∂²/∂x²`, `∂²/∂x∂y`, which reduce to the two
    /// coefficients `(y²−x²)/D²` and `2xy/D²` below (derived once, checked against finite differences
    /// in the crate tests).
    #[must_use]
    pub fn atan2(self, x: Self) -> Self {
        let y = self;
        let d = x.value * x.value + y.value * y.value;
        let inv_d = 1.0 / d;
        let gy = x.value * inv_d; // ∂atan2/∂y
        let gx = -y.value * inv_d; // ∂atan2/∂x
        let c1 = (y.value * y.value - x.value * x.value) * inv_d * inv_d; // ∂gy/∂x = ∂gx/∂y
        let c2 = 2.0 * x.value * y.value * inv_d * inv_d; // ∂gy/∂y = -∂gx/∂x (sign carried below)

        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = gy * y.grad[i] + gx * x.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = gy * y.hess[i][j]
                    + gx * x.hess[i][j]
                    + c1 * (x.grad[i] * y.grad[j] + x.grad[j] * y.grad[i])
                    + c2 * (x.grad[i] * x.grad[j] - y.grad[i] * y.grad[j]);
            }
        }
        Self {
            value: libm::atan2(y.value, x.value),
            grad,
            hess,
        }
    }
}

impl<const N: usize> Add for HyperDual<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut grad = self.grad;
        let mut hess = self.hess;
        for i in 0..N {
            grad[i] += rhs.grad[i];
            for j in 0..N {
                hess[i][j] += rhs.hess[i][j];
            }
        }
        Self {
            value: self.value + rhs.value,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Sub for HyperDual<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut grad = self.grad;
        let mut hess = self.hess;
        for i in 0..N {
            grad[i] -= rhs.grad[i];
            for j in 0..N {
                hess[i][j] -= rhs.hess[i][j];
            }
        }
        Self {
            value: self.value - rhs.value,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Mul for HyperDual<N> {
    type Output = Self;
    /// Product rule to second order: `(uv)' = u'v + uv'`,
    /// `(uv)''ᵢⱼ = u''ᵢⱼ v + u'ᵢ v'ⱼ + u'ⱼ v'ᵢ + u v''ᵢⱼ`.
    fn mul(self, rhs: Self) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = self.grad[i] * rhs.value + self.value * rhs.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = self.hess[i][j] * rhs.value
                    + self.grad[i] * rhs.grad[j]
                    + self.grad[j] * rhs.grad[i]
                    + self.value * rhs.hess[i][j];
            }
        }
        Self {
            value: self.value * rhs.value,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Div for HyperDual<N> {
    type Output = Self;
    /// Quotient rule to second order, from differentiating `w·v = u` twice: with `w = u/v`,
    /// `w'ᵢ = (u'ᵢ − w v'ᵢ)/v` and `w''ᵢⱼ = (u''ᵢⱼ − w'ᵢ v'ⱼ − w'ⱼ v'ᵢ − w v''ᵢⱼ)/v`.
    fn div(self, rhs: Self) -> Self {
        let inv = 1.0 / rhs.value;
        let value = self.value * inv;
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = (self.grad[i] - value * rhs.grad[i]) * inv;
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = (self.hess[i][j]
                    - grad[i] * rhs.grad[j]
                    - grad[j] * rhs.grad[i]
                    - value * rhs.hess[i][j])
                    * inv;
            }
        }
        Self { value, grad, hess }
    }
}

impl<const N: usize> Neg for HyperDual<N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut grad = self.grad;
        let mut hess = self.hess;
        for i in 0..N {
            grad[i] = -grad[i];
            for j in 0..N {
                hess[i][j] = -hess[i][j];
            }
        }
        Self {
            value: -self.value,
            grad,
            hess,
        }
    }
}
