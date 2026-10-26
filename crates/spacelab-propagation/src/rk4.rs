//! Classical fourth-order Runge–Kutta, fixed step — the baseline every other method is checked
//! against.
//!
//! RK4's coefficients are beyond dispute (`b = [1/6, 1/3, 1/3, 1/6]`, `c = [0, 1/2, 1/2, 1]`). It
//! has no error estimate and no dense output; its job is to be obviously correct and to serve as the
//! reference in the convergence-order and differential tests of the adaptive methods.
//!
//! Gate: `derivations/rk4_order.mac` shows in exact rationals that the tableau `rk4_step` forms
//! meets all 8 Butcher order conditions through order four and none of the 9 of order five
//! (re-checked by `derivations/rk4_order.gp` with an independent tree enumeration); the module test
//! `the_shipped_step_meets_every_order_condition_through_four` (G1) reads every one of those
//! conditions off the shipped step through its tree system.

use crate::DynamicalSystem;

/// One fixed step of classical RK4, returning the state at `t + h`.
#[must_use]
pub fn rk4_step<const N: usize>(
    sys: &impl DynamicalSystem<N>,
    t: f64,
    y: &[f64; N],
    h: f64,
) -> [f64; N] {
    let mut k1 = [0.0; N];
    let mut k2 = [0.0; N];
    let mut k3 = [0.0; N];
    let mut k4 = [0.0; N];
    let mut tmp = [0.0; N];

    sys.derivative(t, y, &mut k1);

    for i in 0..N {
        tmp[i] = y[i] + 0.5 * h * k1[i];
    }
    sys.derivative(t + 0.5 * h, &tmp, &mut k2);

    for i in 0..N {
        tmp[i] = y[i] + 0.5 * h * k2[i];
    }
    sys.derivative(t + 0.5 * h, &tmp, &mut k3);

    for i in 0..N {
        tmp[i] = y[i] + h * k3[i];
    }
    sys.derivative(t + h, &tmp, &mut k4);

    let mut out = *y;
    for i in 0..N {
        out[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    out
}

/// Integrate from `t0` to `t1` in `steps` equal RK4 steps, returning the final state.
///
/// Fixed-step, so the caller owns the accuracy/step trade-off. Used as the reference in tests and
/// as the simplest possible propagator.
#[must_use]
pub fn rk4_integrate<const N: usize>(
    sys: &impl DynamicalSystem<N>,
    t0: f64,
    y0: &[f64; N],
    t1: f64,
    steps: u32,
) -> [f64; N] {
    let h = (t1 - t0) / f64::from(steps);
    let mut t = t0;
    let mut y = *y0;
    for _ in 0..steps {
        y = rk4_step(sys, t, &y, h);
        t += h;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::rk4_step;
    use crate::DynamicalSystem;

    /// A rooted tree as the sorted list of its root's subtrees; the empty list is the single vertex.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct Tree(Vec<Tree>);

    impl Tree {
        fn order(&self) -> u32 {
            1 + self.0.iter().map(Tree::order).sum::<u32>()
        }

        /// `γ(t) = ρ(t) ∏ γ(u)` over the root's subtrees.
        fn density(&self) -> f64 {
            f64::from(self.order()) * self.0.iter().map(Tree::density).product::<f64>()
        }
    }

    /// Every rooted tree of order `1..=n`, indexed by order: a root over each multiset of smaller
    /// trees whose orders sum to one less — the enumeration `derivations/rk4_order.mac` uses.
    fn trees(n: u32) -> Vec<Vec<Tree>> {
        fn multisets(
            pool: &[Tree],
            from: usize,
            left: u32,
            cur: &mut Vec<Tree>,
            out: &mut Vec<Tree>,
        ) {
            if left == 0 {
                out.push(Tree(cur.clone()));
                return;
            }
            for i in from..pool.len() {
                let k = pool[i].order();
                if k <= left {
                    cur.push(pool[i].clone());
                    multisets(pool, i, left - k, cur, out);
                    cur.pop();
                }
            }
        }
        let mut by_order = vec![Vec::new(), vec![Tree(Vec::new())]];
        for m in 2..=n {
            let pool: Vec<Tree> = by_order[1..m as usize].iter().flatten().cloned().collect();
            let mut out = Vec::new();
            multisets(&pool, 0, m - 1, &mut Vec::new(), &mut out);
            by_order.push(out);
        }
        by_order
    }

    /// The tree system of a tree: one component per vertex, `y_v' = ∏ y_u` over the children `u` of
    /// `v`, started from `y = 0` at `t = 0`. Its exact solution is `y_v(τ) = τ^{ρ(v)}/γ(v)`. The
    /// system is polynomial and nilpotent, so an explicit Runge–Kutta step on it truncates nothing:
    /// a step of `h = 1` leaves `Σ bᵢ Φᵢ(t)` in the root component exactly, up to rounding. With
    /// `clock` set, every leaf factor is read from the stage time instead of from its component —
    /// `y_leaf(τ) = τ` is the clock itself — so the same condition then also fixes the `t + cᵢ h`
    /// the step passes to the derivative.
    struct TreeSystem {
        children: Vec<Vec<usize>>,
        clock: bool,
    }

    impl TreeSystem {
        fn new(t: &Tree, clock: bool) -> Self {
            fn walk(t: &Tree, children: &mut Vec<Vec<usize>>) -> usize {
                let me = children.len();
                children.push(Vec::new());
                for u in &t.0 {
                    let id = walk(u, children);
                    children[me].push(id);
                }
                me
            }
            let mut children = Vec::new();
            walk(t, &mut children);
            Self { children, clock }
        }
    }

    impl<const N: usize> DynamicalSystem<N> for TreeSystem {
        fn derivative(&self, t: f64, y: &[f64; N], dydt: &mut [f64; N]) {
            *dydt = [0.0; N];
            for (v, kids) in self.children.iter().enumerate() {
                dydt[v] = kids
                    .iter()
                    .map(|&u| {
                        if self.clock && self.children[u].is_empty() {
                            t
                        } else {
                            y[u]
                        }
                    })
                    .product();
            }
        }
    }

    /// (G1) The shipped `rk4_step` meets every Butcher order condition `Σ bᵢ Φᵢ(t) = 1/γ(t)` through
    /// order four and none of order five — the identities `derivations/rk4_order.mac` proves in exact
    /// rationals for the tableau the step forms, re-checked by `derivations/rk4_order.gp`.
    ///
    /// Each tree's condition is read off the shipped step through its tree system, once with the
    /// leaves taken from their components (the row sums of `A`) and once from the stage time (the
    /// nodes `c`). At `h = 1` every stage is a dyadic rational until the final `h/6`, so the order-4
    /// residuals are bounded by a few ulps. The nine order-5 residuals are `±1/80`, `±1/120` and
    /// `±1/240` in the derivation; none is below `1/240`, so the method is not of order five.
    #[test]
    fn the_shipped_step_meets_every_order_condition_through_four() {
        let forest = trees(5);
        let counts: Vec<usize> = forest.iter().map(Vec::len).collect();
        assert_eq!(counts, [0, 1, 1, 2, 4, 9], "rooted trees per order");
        for clock in [false, true] {
            for (order, ts) in forest.iter().enumerate() {
                for t in ts {
                    let y = rk4_step(&TreeSystem::new(t, clock), 0.0, &[0.0; 5], 1.0);
                    let residual = y[0] - 1.0 / t.density();
                    if order <= 4 {
                        assert!(
                            residual.abs() <= 4.0 * f64::EPSILON,
                            "order {order}, clock {clock}: residual {residual:e}"
                        );
                    } else {
                        assert!(
                            residual.abs() >= (1.0 - 1e-12) / 240.0,
                            "order 5, clock {clock}: residual {residual:e}, an order-5 condition holds"
                        );
                    }
                }
            }
        }
    }
}
