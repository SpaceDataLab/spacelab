//! Automatic derivatives against analytic ones, to machine precision — and against finite
//! differences, to show the accuracy AD buys.
//!
//! The Kepler-acceleration Jacobian is the point: it is exactly the force-model partial the state
//! transition matrix and orbit determination need, and it has a closed form to check against. AD
//! matches it to ~1e-13; a finite difference cannot do better than ~1e-8.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::needless_range_loop
)]

use spacelab_autodiff::{Dual, HyperDual, Scalar, hessian, jacobian};
use spacelab_testkit::nan_max;

/// One independent variable, so derivatives are scalars we can compare to calculus.
fn var(x: f64) -> Dual<1> {
    Dual::variable(x, 0)
}
fn d(y: Dual<1>) -> f64 {
    y.grad()[0]
}

#[test]
fn elementary_derivatives_match_calculus() {
    let x = var(1.7);
    // d(x²) = 2x
    assert!((d(x * x) - 2.0 * 1.7).abs() < 1e-15);
    // d(√x) = 1/(2√x)
    assert!((d(x.sqrt()) - 0.5 / 1.7f64.sqrt()).abs() < 1e-15);
    // d(1/x) = -1/x²
    assert!((d(x.recip()) - (-1.0 / (1.7 * 1.7))).abs() < 1e-15);
    // d(sin x) = cos x, d(cos x) = -sin x
    assert!((d(x.sin()) - 1.7f64.cos()).abs() < 1e-15);
    assert!((d(x.cos()) + 1.7f64.sin()).abs() < 1e-15);
    // d(eˣ) = eˣ, d(ln x) = 1/x
    assert!((d(x.exp()) - 1.7f64.exp()).abs() < 1e-15);
    assert!((d(x.ln()) - 1.0 / 1.7).abs() < 1e-15);
    // d(xᵖ) = p·xᵖ⁻¹
    assert!((d(x.powf(2.5)) - 2.5 * 1.7f64.powf(1.5)).abs() < 1e-14);
    assert!((d(x.powi(3)) - 3.0 * 1.7 * 1.7).abs() < 1e-13);
}

#[test]
fn the_chain_rule_composes() {
    let x = var(0.9);
    // d(sin(x²)) = 2x·cos(x²)
    let got = d((x * x).sin());
    let want = 2.0 * 0.9 * (0.9f64 * 0.9).cos();
    assert!((got - want).abs() < 1e-15, "{got} vs {want}");
    // d(exp(√x)) = exp(√x)/(2√x)
    let got = d(x.sqrt().exp());
    let want = (0.9f64.sqrt()).exp() / (2.0 * 0.9f64.sqrt());
    assert!((got - want).abs() < 1e-15);
    // d(ln(1/(1+x²))) = -2x/(1+x²)
    let one = Dual::<1>::constant(1.0);
    let got = d((one / (one + x * x)).ln());
    let want = -2.0 * 0.9 / (1.0 + 0.9 * 0.9);
    assert!((got - want).abs() < 1e-15, "{got} vs {want}");
}

#[test]
fn product_and_quotient_rules_are_exact() {
    // f(x,y) = x²y / (x + y); check both partials against calculus at (2, 3).
    let (fx, jac) = jacobian::<2, 1, _>(
        |v| {
            let (x, y) = (v[0], v[1]);
            [x * x * y / (x + y)]
        },
        &[2.0, 3.0],
    );
    // value
    assert!((fx[0] - (4.0 * 3.0 / 5.0)).abs() < 1e-14);
    // ∂f/∂x = (2xy(x+y) - x²y) / (x+y)²  at (2,3)
    let dfx = (2.0 * 2.0 * 3.0 * 5.0 - 4.0 * 3.0) / 25.0;
    // ∂f/∂y = (x²(x+y) - x²y) / (x+y)² = x³/(x+y)²  at (2,3)
    let dfy = 8.0 / 25.0;
    assert!(
        (jac[0][0] - dfx).abs() < 1e-13,
        "df/dx {} vs {dfx}",
        jac[0][0]
    );
    assert!(
        (jac[1][0] - dfy).abs() < 1e-13,
        "df/dy {} vs {dfy}",
        jac[1][0]
    );
}

/// The two-body acceleration, written once over any [`Scalar`]: `a = -μ r / |r|³`.
fn kepler_accel<S: Scalar>(r: &[S; 3], mu: f64) -> [S; 3] {
    let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    let inv_r3 = r2.powf(-1.5);
    let neg_mu = S::from_f64(-mu);
    [
        neg_mu * r[0] * inv_r3,
        neg_mu * r[1] * inv_r3,
        neg_mu * r[2] * inv_r3,
    ]
}

#[test]
fn the_kepler_acceleration_jacobian_matches_the_closed_form() {
    // This is the force-model partial the STM and orbit determination need. The analytic Jacobian is
    //   ∂aᵢ/∂rⱼ = -μ/|r|³ · (δᵢⱼ - 3 rᵢrⱼ/|r|²).
    let mu = 3.986_004_418e14;
    let positions = [
        [7_000_000.0, 0.0, 0.0],
        [-2_932_912.0, -6_041_930.0, -509_897.0],
        [42_164_000.0, 0.0, 0.0],
        [1_000_000.0, 2_000_000.0, 3_000_000.0],
    ];

    for r in positions {
        // AD: same source as the value, seeded on the 3 position components.
        let (a_val, jac) = jacobian::<3, 3, _>(|v| kepler_accel(v, mu), &r);

        // analytic reference
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let rmag = r2.sqrt();
        let inv_r3 = 1.0 / (r2 * rmag);
        for i in 0..3 {
            // value check
            let a_analytic = -mu * r[i] * inv_r3;
            assert!((a_val[i] - a_analytic).abs() < 1e-6 * a_analytic.abs().max(1.0));
            for j in 0..3 {
                let delta = f64::from(u8::from(i == j));
                let want = -mu * inv_r3 * (delta - 3.0 * r[i] * r[j] / r2);
                let got = jac[j][i]; // jac[input j][output i] = ∂aᵢ/∂rⱼ
                let scale = want.abs().max(mu * inv_r3);
                assert!(
                    (got - want).abs() < 1e-12 * scale,
                    "∂a{i}/∂r{j} at {r:?}: AD {got:.6e} vs analytic {want:.6e}"
                );
            }
        }
    }
}

#[test]
fn autodiff_beats_finite_differences() {
    // The accuracy claim, demonstrated: AD matches the analytic Kepler Jacobian to ~1e-13 relative,
    // while a central finite difference is stuck near √ε ≈ 1e-8 no matter the step.
    let mu = 3.986_004_418e14;
    let r = [7_000_000.0, 1_000_000.0, 500_000.0];
    let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    let rmag = r2.sqrt();
    let inv_r3 = 1.0 / (r2 * rmag);
    let analytic = |i: usize, j: usize| {
        let delta = f64::from(u8::from(i == j));
        -mu * inv_r3 * (delta - 3.0 * r[i] * r[j] / r2)
    };

    // AD error
    let (_, jac) = jacobian::<3, 3, _>(|v| kepler_accel(v, mu), &r);
    let mut ad_worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let rel = (jac[j][i] - analytic(i, j)).abs() / analytic(i, j).abs().max(mu * inv_r3);
            ad_worst = nan_max(ad_worst, rel);
        }
    }

    // best central finite difference over a sweep of steps
    let accel_f64 = |rr: &[f64; 3]| kepler_accel(rr, mu);
    let mut fd_best = f64::INFINITY;
    for p in 1..=12 {
        let h = 10f64.powi(-p) * rmag; // relative step
        let mut worst = 0.0_f64;
        for j in 0..3 {
            let mut rp = r;
            let mut rm = r;
            rp[j] += h;
            rm[j] -= h;
            let ap = accel_f64(&rp);
            let am = accel_f64(&rm);
            for i in 0..3 {
                let fd = (ap[i] - am[i]) / (2.0 * h);
                let rel = (fd - analytic(i, j)).abs() / analytic(i, j).abs().max(mu * inv_r3);
                worst = worst.max(rel);
            }
        }
        fd_best = fd_best.min(worst);
    }

    println!("AD worst relative error {ad_worst:.2e};  best finite-difference {fd_best:.2e}");
    assert!(
        ad_worst < 1e-12,
        "AD should be near machine precision, got {ad_worst:.2e}"
    );
    assert!(
        fd_best > 1e-11,
        "finite differences should not reach AD's accuracy"
    );
    assert!(
        ad_worst < fd_best / 100.0,
        "AD must be far more accurate than finite differences"
    );
}

#[test]
fn atan2_partials_are_correct() {
    // ∂/∂y atan2(y,x) = x/(x²+y²), ∂/∂x = -y/(x²+y²)
    let (_, jac) = jacobian::<2, 1, _>(
        |v| {
            let (y, x) = (v[0], v[1]);
            [y.atan2(x)]
        },
        &[1.5, 2.5],
    );
    let denom = 1.5f64 * 1.5 + 2.5 * 2.5;
    assert!((jac[0][0] - 2.5 / denom).abs() < 1e-15, "d/dy");
    assert!((jac[1][0] - (-1.5 / denom)).abs() < 1e-15, "d/dx");
}

#[test]
fn hyperdual_second_derivatives_match_calculus() {
    // One independent variable, so the Hessian block is the scalar second derivative.
    let x0 = 1.7;
    let x = HyperDual::<1>::variable(x0, 0);
    let d2 = |y: HyperDual<1>| y.hess()[0][0];
    // and the first derivatives still fall out, exactly as for a Dual
    let d1 = |y: HyperDual<1>| y.grad()[0];

    assert!(
        (d1(x * x) - 2.0 * x0).abs() < 1e-14 && (d2(x * x) - 2.0).abs() < 1e-14,
        "x²"
    );
    assert!(
        (d2(x.sqrt()) - (-0.25 / x0.powf(1.5))).abs() < 1e-13,
        "√x: g'' = -¼x^-3/2"
    );
    assert!(
        (d2(x.recip()) - 2.0 / x0.powi(3)).abs() < 1e-13,
        "1/x: g'' = 2/x³"
    );
    // The *first* derivative of each unary rule as well. `x` is a seed variable, so its own Hessian
    // is zero, which annihilates the `d1 · self.hess` term of `chain` — the only place the first
    // derivative reaches a second-order number. Without these, `d1` is unobservable for every one
    // of them.
    assert!(
        (d1(x.sqrt()) - 0.5 / x0.sqrt()).abs() < 1e-14,
        "√x: g' = 1/(2√x)"
    );
    assert!(
        (d1(x.recip()) + 1.0 / (x0 * x0)).abs() < 1e-14,
        "1/x: g' = -1/x²"
    );
    assert!((d1(x.cos()) + x0.sin()).abs() < 1e-14, "cos: g' = -sin");
    assert!((d1(x.sin()) - x0.cos()).abs() < 1e-14, "sin: g' = cos");
    assert!((d1(x.exp()) - x0.exp()).abs() < 1e-13, "eˣ: g' = eˣ");
    assert!((d1(x.ln()) - 1.0 / x0).abs() < 1e-14, "ln x: g' = 1/x");
    assert!(
        (d1(x.powf(2.5)) - 2.5 * x0.powf(1.5)).abs() < 1e-13,
        "xᵖ: g' = p·xᵖ⁻¹"
    );
    // integer power: d(x³) = 3x², d²(x³) = 6x
    assert!(
        (d1(x.powi(3)) - 3.0 * x0 * x0).abs() < 1e-12 && (d2(x.powi(3)) - 6.0 * x0).abs() < 1e-12,
        "x³ via powi"
    );
    // n = 3 is the one exponent where the second-derivative exponent `n - 2` and a truncating
    // `n / 2` agree, so a second one is needed to pin it: 2 − 2 = 0 against 2 / 2 = 1.
    assert!(
        (d1(x.powi(2)) - 2.0 * x0).abs() < 1e-12 && (d2(x.powi(2)) - 2.0).abs() < 1e-12,
        "x² via powi"
    );
    assert!(
        (d2(x.powf(2.5)) - 2.5 * 1.5 * x0.powf(0.5)).abs() < 1e-12,
        "xᵖ: g'' = p(p-1)x^{{p-2}}"
    );
    assert!((d2(x.exp()) - x0.exp()).abs() < 1e-13, "eˣ");
    assert!(
        (d2(x.ln()) - (-1.0 / (x0 * x0))).abs() < 1e-14,
        "ln x: g'' = -1/x²"
    );
    assert!((d2(x.sin()) + x0.sin()).abs() < 1e-14, "sin: g'' = -sin");
    assert!((d2(x.cos()) + x0.cos()).abs() < 1e-14, "cos: g'' = -cos");
    // chain rule to 2nd order: d²/dx² sin(x²) = 2cos(x²) - 4x²sin(x²)
    let want = 2.0 * (x0 * x0).cos() - 4.0 * x0 * x0 * (x0 * x0).sin();
    assert!((d2((x * x).sin()) - want).abs() < 1e-12, "sin(x²)");
    // Div operator, quotient rule to 2nd order: d²/dx² [1/(1+x²)] = (6x²-2)/(1+x²)³
    let one = HyperDual::<1>::constant(1.0);
    let want = (6.0 * x0 * x0 - 2.0) / (1.0 + x0 * x0).powi(3);
    assert!((d2(one / (one + x * x)) - want).abs() < 1e-12, "1/(1+x²)");
}

#[test]
fn the_kepler_acceleration_hessian_matches_the_closed_form() {
    // The force Hessian the STT needs: ∂²aᵢ/∂rⱼ∂rₖ for a = -μr/|r|³. Since a = -∇Φ, this is the (fully
    // symmetric) third derivative of the potential:
    //   Hᵢⱼₖ = 3μ r⁻⁵ (δᵢⱼ rₖ + δᵢₖ rⱼ + δⱼₖ rᵢ) - 15μ r⁻⁷ rᵢ rⱼ rₖ.
    let mu = 3.986_004_418e14;
    let positions = [
        [7_000_000.0, 0.0, 0.0],
        [-2_932_912.0, -6_041_930.0, -509_897.0],
        [1_000_000.0, 2_000_000.0, 3_000_000.0],
    ];
    for r in positions {
        let (_, _, hess) = hessian::<3, 3, _>(|v| kepler_accel(v, mu), &r);
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let inv_r5 = 1.0 / (r2 * r2 * r2.sqrt());
        let inv_r7 = inv_r5 / r2;
        let delta = |a: usize, b: usize| f64::from(u8::from(a == b));
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let want = 3.0
                        * mu
                        * inv_r5
                        * (delta(i, j) * r[k] + delta(i, k) * r[j] + delta(j, k) * r[i])
                        - 15.0 * mu * inv_r7 * r[i] * r[j] * r[k];
                    let scale = want.abs().max(mu * inv_r5 * r2.sqrt());
                    assert!(
                        (hess[i][j][k] - want).abs() < 1e-10 * scale,
                        "∂²a{i}/∂r{j}∂r{k} at {r:?}: AD {:.6e} vs closed form {want:.6e}",
                        hess[i][j][k]
                    );
                    // symmetry to the bit, by construction
                    assert!(
                        hess[i][j][k] == hess[i][k][j],
                        "Hessian must be symmetric in (j,k)"
                    );
                }
            }
        }
    }
}

#[test]
fn hyperdual_atan2_hessian_matches_a_finite_difference_of_the_gradient() {
    // atan2 is the one Scalar op with a hand-derived second-order rule and no closed-form test above.
    // Check its full 2×2 Hessian on f(u) = atan2(u₀², u₁) against a central difference of the
    // AD-exact gradient — which floors near 1e-7, far below any error in the rule itself.
    // Both arguments are non-linear in the seeds. With a bare `v[1]` on the `x` side its Hessian is
    // identically zero, and the `gx · x''` term of the rule — one of the five it is built from —
    // is multiplied away before it reaches anything asserted.
    let f = |v: &[HyperDual<2>; 2]| [(v[0] * v[0]).atan2(v[1] * v[1])];
    let u = [0.7_f64, 1.9_f64];
    let (_, _, hess) = hessian::<2, 1, _>(f, &u);

    let grad_at = |uu: &[f64; 2]| -> [f64; 2] {
        // reuse the value/first-order path by seeding HyperDual and reading its gradient
        let seeds = [
            HyperDual::<2>::variable(uu[0], 0),
            HyperDual::<2>::variable(uu[1], 1),
        ];
        f(&seeds)[0].grad()
    };
    let h = 1e-6;
    let mut worst = 0.0_f64;
    for k in 0..2 {
        let (mut up, mut um) = (u, u);
        up[k] += h;
        um[k] -= h;
        let (gp, gm) = (grad_at(&up), grad_at(&um));
        for j in 0..2 {
            let fd = (gp[j] - gm[j]) / (2.0 * h);
            worst = nan_max(worst, (hess[0][j][k] - fd).abs());
        }
    }
    assert!(
        worst < 1e-6,
        "atan2 Hessian disagrees with the FD-of-gradient by {worst:.3e}"
    );
}
