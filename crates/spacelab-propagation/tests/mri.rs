//! MRI-GARK-ERK33a multirate: a fast rotation coupled to a slow decay, checked at order 3 in the
//! slow macro-step against a fine single-rate reference.

#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    // `integrate_mri` takes `Fn(f64, &[f64; N])`, so a one-component state is passed by reference.
    clippy::trivially_copy_pass_by_ref
)]

use spacelab_propagation::mri::integrate_mri;

const OMEGA: f64 = 10.0; // fast rotation rate
const DECAY: f64 = 0.5; // slow decay rate

fn fast(_t: f64, y: &[f64; 2]) -> [f64; 2] {
    [OMEGA * y[1], -OMEGA * y[0]]
}
fn slow(_t: f64, y: &[f64; 2]) -> [f64; 2] {
    [-DECAY * y[0], -DECAY * y[1]]
}

/// A tight single-rate RK4 reference on the full right-hand side.
fn reference(y0: &[f64; 2], span: f64, steps: usize) -> [f64; 2] {
    let h = span / steps as f64;
    let full = |t: f64, y: &[f64; 2]| {
        let (a, b) = (fast(t, y), slow(t, y));
        [a[0] + b[0], a[1] + b[1]]
    };
    let axpy = |y: &[f64; 2], f: f64, d: &[f64; 2]| [y[0] + f * d[0], y[1] + f * d[1]];
    let mut y = *y0;
    let mut t = 0.0;
    for _ in 0..steps {
        let k1 = full(t, &y);
        let k2 = full(t + h / 2.0, &axpy(&y, h / 2.0, &k1));
        let k3 = full(t + h / 2.0, &axpy(&y, h / 2.0, &k2));
        let k4 = full(t + h, &axpy(&y, h, &k3));
        for k in 0..2 {
            y[k] += h / 6.0 * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
        }
        t += h;
    }
    y
}

fn mri_err(y0: &[f64; 2], span: f64, ref_y: &[f64; 2], slow_steps: usize, m: usize) -> f64 {
    let y = integrate_mri(&fast, &slow, 0.0, y0, span, span / slow_steps as f64, m).unwrap();
    ((y[0] - ref_y[0]).powi(2) + (y[1] - ref_y[1]).powi(2)).sqrt()
}

#[test]
fn mri_converges_at_order_three_in_the_macro_step() {
    let y0 = [1.0, 0.0];
    let span = 2.0;
    let ref_y = reference(&y0, span, 200_000);
    // Many fast sub-steps so the inner solve is effectively exact and the slow coupling sets the order.
    let m = 60;
    let coarse = mri_err(&y0, span, &ref_y, 20, m);
    let fine = mri_err(&y0, span, &ref_y, 40, m);
    let order = (coarse / fine).log2();
    println!("MRI: err {coarse:.3e} → {fine:.3e} (20→40 macro-steps, m={m}), order {order:.2}");
    assert!(
        fine > 1e-12,
        "already at the inner-solve floor; raise m or use coarser grids"
    );
    assert!(
        (order - 3.0).abs() < 0.5,
        "measured order {order:.2} is not 3"
    );
}

#[test]
fn mri_agrees_with_the_single_rate_reference() {
    let y0 = [1.0, 0.0];
    let span = 5.0;
    let ref_y = reference(&y0, span, 500_000);
    let err = mri_err(&y0, span, &ref_y, 100, 80);
    println!("MRI vs single-rate RK4 over span {span}: {err:.3e}");
    assert!(err < 1e-3, "MRI disagrees with the reference: {err:.3e}");
}

// --- the clock, which the autonomous pair above cannot see ------------------------------------

const LAMBDA: f64 = 0.5; // slow decay
const OMEGA_F: f64 = 12.0; // fast forcing, ~3.8 periods across the span
const OMEGA_S: f64 = 0.7; // slow forcing
const AMP_F: f64 = 1.0;
const AMP_S: f64 = 0.6;

/// `y' = −λy + A_F cos(ω_F t) + A_S cos(ω_S t)`, split at the two forcing rates.
fn forced_fast(t: f64, _y: &[f64; 1]) -> [f64; 1] {
    [AMP_F * (OMEGA_F * t).cos()]
}
fn forced_slow(t: f64, y: &[f64; 1]) -> [f64; 1] {
    [-LAMBDA * y[0] + AMP_S * (OMEGA_S * t).cos()]
}

/// The closed form. `y_p(t) = Σ A(λ cos ωt + ω sin ωt)/(λ² + ω²)` is the particular solution of
/// each forcing term; the homogeneous part carries whatever the initial condition leaves over.
fn forced_exact(t0: f64, y0: f64, t: f64) -> f64 {
    let part = |t: f64| {
        [(AMP_F, OMEGA_F), (AMP_S, OMEGA_S)]
            .iter()
            .map(|&(a, w)| {
                a * (LAMBDA * (w * t).cos() + w * (w * t).sin()) / LAMBDA.mul_add(LAMBDA, w * w)
            })
            .sum::<f64>()
    };
    (y0 - part(t0)) * (-LAMBDA * (t - t0)).exp() + part(t)
}

/// MRI-GARK threads a clock through every one of its couplings, and the pair above ignores it.
///
/// `integrate_mri` reads `t` in five places: the macro step `(t1 − t0)/n`, the accumulator
/// `t += h`, the slow-stage abscissae `t + cₛ·h` that set `base_t` and the epoch of `f_S(Yⱼ)`, and
/// the fast-time argument `base_t + Δc·θ` inside the inner RK4. `fast` and `slow` above take `_t`
/// and every call starts at `t0 = 0`, which makes all five unobservable at once — `c₂ = 1/3` may as
/// well be `1` or `3`, `t + c·h` may as well be `t − c·h`, and `t += h` may as well be `t *= h`.
///
/// This pair reads its clock, at two well-separated rates so the fast and slow couplings are
/// exercised for what they are, and runs from `t0 = 1`. Measured relative error against the closed
/// form: 2.2171e-3, 2.4082e-4, 2.8920e-5 and 3.5672e-6 at 10, 20, 40 and 80 macro-steps — ratios
/// 9.21, 8.33 and 8.11, the third order the method claims.
#[test]
fn mri_reads_the_clock_it_threads_through_its_couplings() {
    let (t0, t1, y0) = (1.0_f64, 3.0_f64, 0.4_f64);
    let exact = forced_exact(t0, y0, t1);
    let m = 40;
    let rel = |slow_steps: usize| {
        let h = (t1 - t0) / slow_steps as f64;
        let y = integrate_mri(&forced_fast, &forced_slow, t0, &[y0], t1, h, m).unwrap();
        ((y[0] - exact) / exact).abs()
    };

    let (coarse, fine) = (rel(40), rel(80));
    let order = (coarse / fine).log2();
    println!(
        "MRI on a forced system: {coarse:.4e} -> {fine:.4e} (40->80 macro-steps), order {order:.2}"
    );
    assert!(
        (2.7..3.3).contains(&order),
        "MRI measured order {order:.2} is not 3 on a system that reads t: {coarse:.3e} -> {fine:.3e}"
    );
    assert!(
        (3.2e-6..4.0e-6).contains(&fine),
        "MRI relative error {fine:.4e} at 80 macro-steps is not the measured 3.567e-6"
    );
}

#[test]
fn mri_rejects_bad_arguments() {
    let y0 = [1.0, 0.0];
    assert!(integrate_mri(&fast, &slow, 0.0, &y0, -1.0, 0.1, 10).is_err());
    assert!(integrate_mri(&fast, &slow, 0.0, &y0, 1.0, 0.0, 10).is_err());
    assert!(integrate_mri(&fast, &slow, 0.0, &y0, 1.0, 0.1, 0).is_err());
}
