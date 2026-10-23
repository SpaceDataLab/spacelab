//! The `Scalar` trait's per-type delegations and the `Dual` operators the calculus tests don't reach.
//!
//! `derivatives.rs` verifies the transcendental *derivatives*; it never calls the trait methods
//! polymorphically (so the `Scalar for f64` / `Dual` / `HyperDual` delegations went unexercised) nor
//! the affine operators `abs`, `Sub`, `Neg` and the scalar-rhs `+ - * /`. This closes both.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_autodiff::{Dual, HyperDual, Scalar};

/// Every `Scalar` method, called through the trait (UFCS), returning the ten resulting values with
/// derivatives dropped. Instantiating this at `T` exercises the whole `impl Scalar for T`.
fn scalar_values<S: Scalar>(x: f64, p: f64) -> [f64; 10] {
    let s = S::from_f64(x);
    [
        Scalar::value(s),
        S::sqrt(s).value(),
        S::recip(s).value(),
        S::powf(s, p).value(),
        S::sin(s).value(),
        S::cos(s).value(),
        S::exp(s).value(),
        S::ln(s).value(),
        S::atan2(s, S::from_f64(1.3)).value(),
        (s + s).value(),
    ]
}

#[test]
fn the_scalar_trait_delegates_consistently_across_f64_dual_and_hyperdual() {
    let (x, p) = (0.7, 1.7);
    // f64 is the reference: each entry is the corresponding libm/std value.
    let want = [
        x,
        x.sqrt(),
        x.recip(),
        x.powf(p),
        x.sin(),
        x.cos(),
        x.exp(),
        x.ln(),
        x.atan2(1.3),
        2.0 * x,
    ];
    for (impl_name, got) in [
        ("f64", scalar_values::<f64>(x, p)),
        ("Dual", scalar_values::<Dual<1>>(x, p)),
        ("HyperDual", scalar_values::<HyperDual<1>>(x, p)),
    ] {
        for i in 0..10 {
            assert!(
                (got[i] - want[i]).abs() <= 1e-15 * want[i].abs().max(1.0),
                "{impl_name} entry {i}: {} != {}",
                got[i],
                want[i],
            );
        }
    }
}

#[test]
#[allow(clippy::float_cmp)] // the operators are exact affine maps on these inputs
fn dual_operators_and_constructors_carry_the_right_derivative() {
    // new(): an explicit value and gradient.
    let a = Dual::<2>::new(2.0, [3.0, 4.0]);
    assert_eq!(a.value(), 2.0);
    assert_eq!(a.grad(), [3.0, 4.0]);
    let b = Dual::<2>::new(5.0, [1.0, -2.0]);

    // Dual − Dual is componentwise; Neg flips value and gradient.
    assert_eq!((a - b).value(), -3.0);
    assert_eq!((a - b).grad(), [2.0, 6.0]);
    assert_eq!((-a).value(), -2.0);
    assert_eq!((-a).grad(), [-3.0, -4.0]);

    // Scalar-rhs operators: a constant shift leaves the slope, a scale multiplies it.
    assert_eq!((a + 10.0).value(), 12.0);
    assert_eq!((a + 10.0).grad(), a.grad());
    assert_eq!((a - 10.0).value(), -8.0);
    assert_eq!((a - 10.0).grad(), a.grad());
    assert_eq!((a * 3.0).value(), 6.0);
    assert_eq!((a * 3.0).grad(), [9.0, 12.0]);
    assert_eq!((a / 2.0).value(), 1.0);
    assert_eq!((a / 2.0).grad(), [1.5, 2.0]);

    // abs(): the derivative is the sign of the value (tested away from the kink at 0).
    let pos = Dual::<1>::variable(3.0, 0);
    assert_eq!(pos.abs().value(), 3.0);
    assert_eq!(pos.abs().grad(), [1.0]);
    let neg = Dual::<1>::variable(-3.0, 0);
    assert_eq!(neg.abs().value(), 3.0);
    assert_eq!(neg.abs().grad(), [-1.0]);
    // And at the kink itself, where the documented derivative is 0 and neither sign is right.
    // Both signed zeros, because `-0.0 >= 0.0` and `-0.0 <= 0.0` are both true in IEEE-754, so a
    // guard widened on either side would pick a sign here.
    assert_eq!(Dual::<1>::variable(0.0, 0).abs().grad(), [0.0]);
    assert_eq!(Dual::<1>::variable(-0.0, 0).abs().grad(), [0.0]);
}

#[test]
fn hyperdual_sub_and_neg_propagate_value_gradient_and_hessian() {
    // The Hessian tests exercise Mul and Div; Sub and Neg had no test.
    let x0 = 1.3;
    let x = HyperDual::<1>::variable(x0, 0);

    // f = x² − x  ⇒  f = x0² − x0,  f′ = 2x0 − 1,  f″ = 2.
    let f = x * x - x;
    assert!((f.value() - (x0 * x0 - x0)).abs() < 1e-14);
    assert!((f.grad()[0] - (2.0 * x0 - 1.0)).abs() < 1e-13);
    assert!((f.hess()[0][0] - 2.0).abs() < 1e-13);

    // The subtrahend above is a bare seed, so its Hessian is zero and `Sub`'s second-order loop
    // subtracts nothing. h = x² − x³ ⇒ h′ = 2x0 − 3x0², h″ = 2 − 6x0.
    let h = x * x - x * x * x;
    assert!((h.value() - (x0 * x0 - x0 * x0 * x0)).abs() < 1e-14);
    assert!((h.grad()[0] - (2.0 * x0 - 3.0 * x0 * x0)).abs() < 1e-13);
    assert!((h.hess()[0][0] - (2.0 - 6.0 * x0)).abs() < 1e-13);

    // g = −(x²)  ⇒  g = −x0²,  g′ = −2x0,  g″ = −2.
    let g = -(x * x);
    assert!((g.value() + x0 * x0).abs() < 1e-14);
    assert!((g.grad()[0] + 2.0 * x0).abs() < 1e-13);
    assert!((g.hess()[0][0] + 2.0).abs() < 1e-13);
}
