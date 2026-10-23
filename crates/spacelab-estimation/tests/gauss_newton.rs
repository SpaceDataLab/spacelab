//! Gauss–Newton with Levenberg–Marquardt: recovers the true parameters of a nonlinear fit, and
//! converges from a poor guess where undamped Gauss–Newton would not.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::cast_precision_loss
)]

use spacelab_estimation::{GaussNewton, ResidualModel};

/// Fit `y = a · exp(b · t)` — the classic two-parameter nonlinear least-squares problem.
struct ExpFit {
    t: [f64; 11],
    y: [f64; 11],
}

impl ExpFit {
    /// Synthetic data from the true parameters, no noise, so the fit must recover them exactly.
    fn from_truth(a: f64, b: f64) -> Self {
        let mut t = [0.0; 11];
        let mut y = [0.0; 11];
        for k in 0..11 {
            t[k] = 0.2 * k as f64;
            y[k] = a * (b * t[k]).exp();
        }
        Self { t, y }
    }
}

impl ResidualModel<2> for ExpFit {
    fn residuals(&self, x: &[f64; 2], emit: &mut dyn FnMut(&[f64; 2], f64, f64)) {
        let (a, b) = (x[0], x[1]);
        for k in 0..11 {
            let e = (b * self.t[k]).exp();
            let residual = a * e - self.y[k];
            let jac = [e, a * self.t[k] * e]; // ∂r/∂a, ∂r/∂b
            emit(&jac, residual, 1.0);
        }
    }
}

/// A linear model, to confirm the Gauss–Newton step is exact in one iteration (no damping).
struct LinearModel {
    rows: [[f64; 3]; 6],
    z: [f64; 6],
}

impl ResidualModel<3> for LinearModel {
    fn residuals(&self, x: &[f64; 3], emit: &mut dyn FnMut(&[f64; 3], f64, f64)) {
        for k in 0..6 {
            let pred = self.rows[k][0] * x[0] + self.rows[k][1] * x[1] + self.rows[k][2] * x[2];
            emit(&self.rows[k], pred - self.z[k], 1.0);
        }
    }
}

/// Three inconsistent measurements of one scalar: `x = 1`, `x = 2`, `x = 6`.
///
/// The least-squares answer is their mean, 3, with a residual cost of 14 that no step can reduce.
/// Every quantity the stopping rule looks at is then exact small-integer arithmetic: from `x₀ = 4`
/// the undamped step is `−1`, the relative step norm `1/4`, the cost `17 → 14` and the relative
/// drop `3/17`. A cost that cannot reach zero lets the absolute floor be set to zero, so each
/// criterion can be isolated by zeroing the other two.
struct ScalarMean;

impl ResidualModel<1> for ScalarMean {
    fn residuals(&self, x: &[f64; 1], emit: &mut dyn FnMut(&[f64; 1], f64, f64)) {
        for z in [1.0, 2.0, 6.0] {
            emit(&[1.0], x[0] - z, 1.0);
        }
    }
}

/// Only the step criterion armed, with the tolerance placed either side of the step's known norm.
///
/// The relative step norm is `|−1| / max(|4|, 1) = 0.25`, so a tolerance of 0.5 must stop the solve
/// on the first iteration and one of 0.2 must not. That brackets the quantity itself rather than
/// the fact that a solve terminates: dividing by the state norm instead of multiplying, taking a
/// remainder, or summing the squares wrongly all move it past one bound or the other.
#[test]
fn the_step_criterion_fires_on_the_relative_norm_of_the_step_and_not_on_something_near_it() {
    let armed = |step_tolerance| GaussNewton {
        step_tolerance,
        cost_tolerance: 0.0,
        absolute_cost_floor: 0.0,
        initial_lambda: 0.0,
        max_iterations: 20,
    };

    let (x, _cov, report) = armed(0.5).solve(&ScalarMean, &[4.0]).expect("solved");
    assert_eq!(
        report.iterations, 1,
        "a step of relative norm 0.25 is below a tolerance of 0.5 and nothing else can stop this \
         solve, so it must stop here: {report:?}"
    );
    assert!(report.converged, "{report:?}");
    assert!((x[0] - 3.0).abs() < 1e-12, "x = {}", x[0]);

    let (_x, _cov, report) = armed(0.2).solve(&ScalarMean, &[4.0]).expect("solved");
    assert!(
        report.iterations >= 2,
        "0.25 is not below 0.2, so the first iteration must not have stopped it: {report:?}"
    );

    // The field says "below this", and 0.25 is not below 0.25. Both operands are exact here — the
    // step is −1 and the state norm 4 — so the boundary itself is reachable, not a rounding story.
    let (_x, _cov, report) = armed(0.25).solve(&ScalarMean, &[4.0]).expect("solved");
    assert!(
        report.iterations >= 2,
        "the step tolerance is exclusive: {report:?}"
    );
}

/// Only the cost criterion armed, with the tolerance either side of the relative drop.
///
/// The cost falls 17 → 14 on the first step, a relative drop of `3/17 = 0.17647…`. A tolerance of
/// 0.2 must stop the solve there and one of 0.1 must not — which pins the drop as a *ratio* of a
/// *difference*, the two operations the survivors rewrite.
#[test]
fn the_cost_criterion_fires_on_the_relative_drop_and_not_on_something_near_it() {
    let armed = |cost_tolerance| GaussNewton {
        step_tolerance: 0.0,
        cost_tolerance,
        absolute_cost_floor: 0.0,
        initial_lambda: 0.0,
        max_iterations: 20,
    };

    let (_x, _cov, report) = armed(0.2).solve(&ScalarMean, &[4.0]).expect("solved");
    assert_eq!(
        report.iterations, 1,
        "a drop of 3/17 is below a tolerance of 0.2: {report:?}"
    );
    assert!(report.converged, "{report:?}");

    let (_x, _cov, report) = armed(0.1).solve(&ScalarMean, &[4.0]).expect("solved");
    assert!(
        report.iterations >= 2,
        "3/17 is 0.176, not below 0.1: {report:?}"
    );

    // Same boundary, same reason: the costs are exactly 17 and 14, so the solver's quotient is the
    // same two f64 operands divided the same way, and "falls by less than this" excludes equality.
    let (_x, _cov, report) = armed(3.0 / 17.0)
        .solve(&ScalarMean, &[4.0])
        .expect("solved");
    assert!(
        report.iterations >= 2,
        "the cost tolerance is exclusive: {report:?}"
    );
}

/// The cap is a cap on the iterations the report counts.
///
/// With every tolerance at zero nothing can converge, so the solve runs to `max_iterations` and
/// says so. The count is the observable: one more would mean the loop ran once past its bound.
#[test]
fn a_solve_that_cannot_converge_stops_at_the_cap_and_reports_it() {
    let gn = GaussNewton {
        step_tolerance: 0.0,
        cost_tolerance: 0.0,
        absolute_cost_floor: 0.0,
        initial_lambda: 1e-3,
        max_iterations: 4,
    };
    let (_x, _cov, report) = gn.solve(&ScalarMean, &[4.0]).expect("solved");
    assert_eq!(report.iterations, 4, "{report:?}");
    assert!(!report.converged, "{report:?}");
}

/// Starting at the answer, the damping schedule is the only thing that ends the solve.
///
/// `ScalarMean`'s minimum is `x = 3` with a cost of 14 that nothing can reduce, so from `x₀ = 3`
/// every iteration takes the rejection branch: the step is zero, the trial cost equals the cost, and
/// `trial_cost < cost` is false forever. With every tolerance zeroed, the *only* exit that reports
/// convergence is the damping saturating at `λ ≥ 1e12`, which is what makes this a test of the
/// schedule's shape rather than of its speed.
///
/// Geometric growth gets there in 15 steps from `λ₀ = 1e-3`. Arithmetic growth of the same
/// increment would need 1e11, so the same solver would run out of iterations and report
/// `converged: false` — a caller re-running an already-converged fit would be told its answer did
/// not converge. Bounding the iterations from *above* is what distinguishes the two; asserting
/// merely that the solve terminates does not.
#[test]
fn a_solve_that_starts_at_the_answer_saturates_the_damping_and_reports_convergence() {
    let gn = GaussNewton {
        step_tolerance: 0.0,
        cost_tolerance: 0.0,
        absolute_cost_floor: 0.0,
        initial_lambda: 1e-3,
        max_iterations: 50,
    };
    let (x, _cov, report) = gn.solve(&ScalarMean, &[3.0]).expect("solved");
    // Exact by construction rather than by luck: the accept branch is the only writer of `x`, and it
    // is never taken here, so the returned state is the bits that went in.
    #[allow(clippy::float_cmp)]
    let unmoved = x[0] == 3.0;
    assert!(unmoved, "the minimum should not have moved: {report:?}");
    assert!(
        report.converged,
        "no step can improve a solve that starts at the minimum, so the damping has to saturate \
         and call that convergence: {report:?}"
    );
    assert!(
        report.iterations <= 16,
        "reaching λ = 1e12 from 1e-3 by decades takes 15 rejections, so 16 is the cap with a step \
         to spare. {} means the schedule is not multiplying: {report:?}",
        report.iterations
    );
}

#[test]
fn a_linear_problem_converges_in_one_step() {
    let x_true = [2.0, -3.0, 1.5];
    let rows = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ];
    let mut z = [0.0; 6];
    for k in 0..6 {
        z[k] = rows[k][0] * x_true[0] + rows[k][1] * x_true[1] + rows[k][2] * x_true[2];
    }
    let model = LinearModel { rows, z };
    let gn = GaussNewton {
        initial_lambda: 0.0, // pure Gauss–Newton
        ..GaussNewton::default()
    };
    let (x, _cov, report) = gn.solve(&model, &[0.0, 0.0, 0.0]).expect("solved");
    assert!(report.converged);
    assert!(
        report.iterations <= 2,
        "linear should converge almost immediately, took {}",
        report.iterations
    );
    for i in 0..3 {
        assert!(
            (x[i] - x_true[i]).abs() < 1e-12,
            "x[{i}] = {} vs {}",
            x[i],
            x_true[i]
        );
    }
    assert!(report.cost < 1e-20, "noiseless fit should have ~zero cost");
}

#[test]
fn recovers_the_true_parameters_of_an_exponential_fit() {
    let (a_true, b_true) = (2.5, -0.7);
    let model = ExpFit::from_truth(a_true, b_true);
    let gn = GaussNewton::default();
    let (x, cov, report) = gn.solve(&model, &[1.0, -0.1]).expect("solved");
    assert!(report.converged, "should converge: {report:?}");
    assert!((x[0] - a_true).abs() < 1e-7, "a = {} vs {a_true}", x[0]);
    assert!((x[1] - b_true).abs() < 1e-7, "b = {} vs {b_true}", x[1]);
    // covariance is symmetric positive (a real fit)
    assert!(cov[0][0] > 0.0 && cov[1][1] > 0.0 && (cov[0][1] - cov[1][0]).abs() < 1e-12);
}

#[test]
fn levenberg_marquardt_converges_from_a_poor_initial_guess() {
    // A far, wrong-sign-ish guess where a raw Gauss–Newton step can overshoot; LM damping must pull
    // it back and still reach the true parameters.
    let (a_true, b_true) = (3.0, 0.4);
    let model = ExpFit::from_truth(a_true, b_true);
    let gn = GaussNewton {
        initial_lambda: 1.0,
        max_iterations: 200,
        ..GaussNewton::default()
    };
    let (x, _cov, report) = gn.solve(&model, &[0.1, 2.0]).expect("solved");
    assert!(
        report.converged,
        "LM should converge from a poor guess: {report:?}"
    );
    assert!((x[0] - a_true).abs() < 1e-5, "a = {} vs {a_true}", x[0]);
    assert!((x[1] - b_true).abs() < 1e-5, "b = {} vs {b_true}", x[1]);
}

#[test]
fn a_noisy_fit_has_a_positive_cost_and_a_finite_covariance() {
    // Perturb one data point; the fit no longer passes through everything, so cost > 0, and the
    // covariance quantifies the remaining uncertainty.
    let mut model = ExpFit::from_truth(2.0, -0.5);
    model.y[5] += 0.05; // an outlier
    let gn = GaussNewton::default();
    let (_x, cov, report) = gn.solve(&model, &[1.0, -0.1]).expect("solved");
    assert!(report.cost > 0.0, "a perturbed fit must have residual cost");
    assert!(cov[0][0].is_finite() && cov[0][0] > 0.0);
}

/// The same fit with the parameters expressed in different units: `x` here is `x/scale` there.
///
/// A scaling of the state is the case Marquardt's `λ·diag(HᵀWH)` exists for. Under it the damped
/// normal equations become `S(Λ + λ diag Λ)S · Δx' = S·(−HᵀWr)`, so `Δx' = S⁻¹Δx` exactly and the
/// whole iteration is the same iteration in different units. Under Levenberg's `λI` it is not:
/// `λ` is then compared against `Λ`'s own entries, which the rescaling has moved.
struct RescaledExpFit {
    inner: ExpFit,
    scale: [f64; 2],
}

impl ResidualModel<2> for RescaledExpFit {
    fn residuals(&self, x: &[f64; 2], emit: &mut dyn FnMut(&[f64; 2], f64, f64)) {
        let unscaled = [x[0] * self.scale[0], x[1] * self.scale[1]];
        self.inner.residuals(&unscaled, &mut |h, r, w| {
            emit(&[h[0] * self.scale[0], h[1] * self.scale[1]], r, w);
        });
    }
}

#[test]
fn the_damping_does_not_notice_a_rescaling_of_the_state() {
    let (a_true, b_true) = (3.0, 0.4);
    let start = [0.1, 2.0];
    let gn = GaussNewton {
        initial_lambda: 1.0,
        max_iterations: 200,
        ..GaussNewton::default()
    };
    let (_x, _cov, base) = gn
        .solve(&ExpFit::from_truth(a_true, b_true), &start)
        .expect("solved");

    for scale in [[1e-6, 1.0], [1.0, 1e5], [1e4, 1e-3], [1e-8, 1e8]] {
        let model = RescaledExpFit {
            inner: ExpFit::from_truth(a_true, b_true),
            scale,
        };
        let (x, _cov, report) = gn
            .solve(&model, &[start[0] / scale[0], start[1] / scale[1]])
            .expect("solved");
        assert_eq!(
            report.iterations, base.iterations,
            "scale {scale:?} changed the iteration count: {report:?} vs {base:?}"
        );
        assert!(
            (x[0] * scale[0] - a_true).abs() < 1e-5 && (x[1] * scale[1] - b_true).abs() < 1e-5,
            "scale {scale:?} recovered {:?}",
            [x[0] * scale[0], x[1] * scale[1]]
        );
    }
}

#[test]
fn the_damping_does_not_notice_a_rescaling_of_the_weights() {
    // The other half of the same statement: the weights carry `1/σ²`, so choosing metres over
    // kilometres multiplies every weight by 10⁶ and every entry of `HᵀWH` with it. `λ·diag` divides
    // that factor straight back out; `λI` does not, and the schedule spends iterations climbing.
    struct Reweighted(ExpFit, f64);
    impl ResidualModel<2> for Reweighted {
        fn residuals(&self, x: &[f64; 2], emit: &mut dyn FnMut(&[f64; 2], f64, f64)) {
            self.0.residuals(x, &mut |h, r, w| emit(h, r, w * self.1));
        }
    }

    // `absolute_cost_floor` is switched off, and the factors are powers of two. Both are about
    // isolating the damping: the floor is an absolute number in cost units and so moves with the
    // weights whatever the damping does, and an exactly representable factor scales every
    // intermediate by an exponent alone, so the two runs agree bit for bit or not at all.
    let gn = GaussNewton {
        initial_lambda: 1.0,
        max_iterations: 200,
        absolute_cost_floor: 0.0,
        ..GaussNewton::default()
    };
    let start = [0.1, 2.0];
    let (x0, _cov, base) = gn
        .solve(&ExpFit::from_truth(3.0, 0.4), &start)
        .expect("solved");

    for factor in [f64::powi(2.0, -30), 0.25, 4.0, f64::powi(2.0, 30)] {
        let model = Reweighted(ExpFit::from_truth(3.0, 0.4), factor);
        let (x, _cov, report) = gn.solve(&model, &start).expect("solved");
        assert_eq!(
            report.iterations, base.iterations,
            "weight factor {factor:e} changed the iteration count: {report:?} vs {base:?}"
        );
        assert!(
            (x[0] - x0[0]).abs() < 1e-9 && (x[1] - x0[1]).abs() < 1e-9,
            "weight factor {factor:e} moved the solution to {x:?} from {x0:?}"
        );
    }
}
