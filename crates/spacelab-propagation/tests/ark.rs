//! ARK4(3)6L[2]SA IMEX: order 4 on the Prothero–Robinson problem, stability when the implicit part
//! is arbitrarily stiff (the whole reason for the implicit–explicit split), and convergence of the
//! stage solve on an implicit part that is nonlinear and does not vanish on the solution.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_autodiff::Scalar;
use spacelab_propagation::ark::integrate_ark;
use spacelab_propagation::variational::ScalarSystem;

/// Prothero–Robinson `y' = −λ(y − sin t) + cos t`, split into the stiff implicit part `f_I = −λ(y − sin t)`
/// and the non-stiff explicit part `f_E = cos t`. Exact solution with `y(0)=0` is `y(t)=sin t`.
struct Stiff {
    lambda: f64,
}
impl ScalarSystem<1> for Stiff {
    fn eval<S: Scalar>(&self, t: S, y: &[S; 1]) -> [S; 1] {
        [(y[0] - t.sin()) * S::from_f64(-self.lambda)]
    }
}
struct Slow;
impl ScalarSystem<1> for Slow {
    fn eval<S: Scalar>(&self, t: S, _y: &[S; 1]) -> [S; 1] {
        [t.cos()]
    }
}

/// A stiff term the stage solve cannot shortcut. Prothero–Robinson's `f_I` is linear *and* vanishes
/// on the exact solution, so the frozen-Jacobian Newton starts at the answer and one step is exact;
/// `f_I = −λy²` is neither, so the iteration has to converge on its own. Paired with
/// [`Manufactured`] the exact solution is `y(t) = A(2 + sin t)`.
struct QuadraticSink {
    lambda: f64,
}
impl ScalarSystem<1> for QuadraticSink {
    fn eval<S: Scalar>(&self, _t: S, y: &[S; 1]) -> [S; 1] {
        [y[0] * y[0] * S::from_f64(-self.lambda)]
    }
}

/// The forcing `f_E = g′(t) + λg(t)²` that makes `g(t) = A(2 + sin t)` an exact solution of
/// `y' = f_E(t) + f_I(y)` with [`QuadraticSink`]'s `f_I`: at `y = g` the two `λg²` cancel and `g′`
/// is left.
struct Manufactured {
    lambda: f64,
    amplitude: f64,
}
impl Manufactured {
    fn solution<S: Scalar>(&self, t: S) -> S {
        S::from_f64(self.amplitude) * (S::from_f64(2.0) + t.sin())
    }
}
impl ScalarSystem<1> for Manufactured {
    fn eval<S: Scalar>(&self, t: S, _y: &[S; 1]) -> [S; 1] {
        let g = self.solution(t);
        [S::from_f64(self.amplitude) * t.cos() + S::from_f64(self.lambda) * g * g]
    }
}

fn err_at(lambda: f64, span: f64, steps: usize) -> f64 {
    let y = integrate_ark::<1>(
        &Slow,
        &Stiff { lambda },
        0.0,
        &[0.0],
        span,
        span / steps as f64,
    )
    .expect("ark");
    (y[0] - span.sin()).abs()
}

#[test]
fn ark_converges_at_order_four() {
    // Moderate stiffness so truncation, not the Newton solve, dominates.
    let lambda = 5.0;
    let span = 4.0;
    let coarse = err_at(lambda, span, 40);
    let fine = err_at(lambda, span, 80);
    let order = (coarse / fine).log2();
    println!("ARK: err {coarse:.3e} → {fine:.3e} (40→80 steps), order {order:.2}");
    assert!(fine > 1e-13, "already at roundoff; use coarser grids");
    assert!(
        (order - 4.0).abs() < 0.5,
        "measured order {order:.2} is not 4"
    );
}

#[test]
fn ark_stays_accurate_when_the_implicit_part_is_very_stiff() {
    // λ = 1e6: an explicit method would need λh ≲ 3 (millions of steps). The ESDIRK on the stiff part
    // is L-stable, so a modest step still tracks sin t — the point of IMEX.
    let lambda = 1.0e6;
    let span = std::f64::consts::TAU;
    let steps = 200; // λh ≈ 3×10⁴ ≫ any explicit stability limit
    let y = integrate_ark::<1>(
        &Slow,
        &Stiff { lambda },
        0.0,
        &[0.0],
        span,
        span / f64::from(steps),
    )
    .expect("ark");
    let err = (y[0] - span.sin()).abs();
    println!(
        "ARK stiff λ=1e6, {steps} steps: y = {:.6}, sin(2π) = {:.6}, err {err:.3e}",
        y[0],
        span.sin()
    );
    assert!(
        y[0].is_finite() && err < 1e-3,
        "IMEX not stable/accurate on the stiff part: {err:.3e}"
    );
}

#[test]
fn ark_solves_a_nonlinear_stage_to_convergence() {
    // A = 1e6 lifts ‖y‖ far above 1, where the Newton test `‖Δ‖ ≤ tol·(1 + ‖y‖)` is a *relative*
    // one: an absolute 1e-12 could never be met at this magnitude. λ = 1e-6 puts λy² alongside g′,
    // so the stage residual is genuinely nonlinear and the frozen-Jacobian iteration needs ~4 steps.
    // Truncating it at one leaves an O(‖Δ‖·ρ) stage error that no later stage repairs.
    let amplitude = 1.0e6;
    let lambda = 1.0e-6;
    let explicit = Manufactured { lambda, amplitude };
    let implicit = QuadraticSink { lambda };
    let (t0, t1) = (1.0, 3.0);
    let y = integrate_ark::<1>(
        &explicit,
        &implicit,
        t0,
        &[explicit.solution(t0)],
        t1,
        0.005,
    )
    .expect("ark");
    let exact = explicit.solution(t1);
    let rel = ((y[0] - exact) / exact).abs();
    println!(
        "ARK nonlinear f_I: y = {:.9e}, exact = {exact:.9e}, rel err {rel:.3e}",
        y[0]
    );
    assert!(rel < 1.0e-9, "stage solve not converged: rel err {rel:.3e}");
}

#[test]
fn ark_grids_the_span_and_not_the_endpoints() {
    // Same span and step as `ark_converges_at_order_four`, moved off the time origin: the step count
    // and the step size must come from t1 − t0. At t0 = 0 the sum of the endpoints and their
    // difference agree, and any grid built from the wrong one still lands on the right answer.
    let t0 = -3.0;
    let t1 = 1.0;
    let y =
        integrate_ark::<1>(&Slow, &Stiff { lambda: 5.0 }, t0, &[t0.sin()], t1, 0.05).expect("ark");
    let err = (y[0] - t1.sin()).abs();
    println!(
        "ARK on [{t0}, {t1}]: y = {:.9}, sin = {:.9}, err {err:.3e}",
        y[0],
        t1.sin()
    );
    assert!(
        err < 1.0e-5,
        "span not integrated on its own grid: err {err:.3e}"
    );
}

#[test]
fn ark_rejects_bad_arguments() {
    assert!(integrate_ark::<1>(&Slow, &Stiff { lambda: 1.0 }, 0.0, &[0.0], -1.0, 0.1).is_err());
    assert!(integrate_ark::<1>(&Slow, &Stiff { lambda: 1.0 }, 0.0, &[0.0], 1.0, 0.0).is_err());
    // An empty span has no grid: `.max(1.0)` would otherwise silently take one step of size zero.
    assert!(integrate_ark::<1>(&Slow, &Stiff { lambda: 1.0 }, 2.0, &[0.0], 2.0, 0.1).is_err());
}

#[test]
fn the_stage_solve_keeps_an_absolute_floor_under_its_convergence_test() {
    // The Newton test is `‖Δ‖ ≤ tol·(1 + ‖y‖)` — relative above ‖y‖ = 1, absolute below it. Every
    // fixture above sits at one of the two magnitudes where the `1 +` cannot be seen: `Stiff` tracks
    // sin t, so ‖y‖ ≈ 1 and the floor is worth a factor of two, and the nonlinear fixture runs at
    // A = 1e6, where it is swamped. Drop the amplitude to 1e-3 and the floor *is* the bound: without
    // it the test is 500× tighter, the frozen-Jacobian iteration cannot reach that inside
    // `MAX_NEWTON`, and the stage solve fails outright rather than returning a worse answer.
    //
    // Measured here: rel err 1.7366e-4 with the floor, `StageSolveFailed` without it. λ = 5e5 is one
    // factor of two inside the method's own convergence limit on this problem (λ = 1e6 fails either
    // way) — the widest separation the cliff allows.
    let amplitude = 1.0e-3;
    let lambda = 5.0e5;
    let explicit = Manufactured { lambda, amplitude };
    let implicit = QuadraticSink { lambda };
    let (t0, t1) = (1.0, 3.0);
    let y = integrate_ark::<1>(
        &explicit,
        &implicit,
        t0,
        &[explicit.solution(t0)],
        t1,
        0.005,
    )
    .expect("the stage solve did not converge at a state norm far below one");
    let exact = explicit.solution(t1);
    let rel = ((y[0] - exact) / exact).abs();
    println!("ARK at amplitude {amplitude:.0e}, lambda {lambda:.0e}: rel err {rel:.4e}");
    assert!(
        rel < 1.0e-3,
        "stage solve converged to the wrong point: {rel:.3e}"
    );
}
