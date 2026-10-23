//! Distributed consensus over inter-satellite links that are not instantaneous.
//!
//! The workspace can *build* a formation — Walker, street-of-coverage, Flower, Draim,
//! kinematically-regular — and score its coverage. Nothing controlled one. A distributed formation
//! controller has no central node: each satellite acts on its neighbours' states as the links
//! delivered them, which is not as they are now, and whether the formation converges at all is a
//! property of the graph, the gains and that delay together.
//!
//! # Two protocols, and they do not have the same answer
//!
//! Write the communication graph as a fixed connected undirected graph with non-negative link gains
//! `a_ij`, degree matrix `D = diag(Σⱼ a_ij)`, adjacency `A`, and Laplacian `L = D − A`. Two things
//! an agent can plausibly do with a late message:
//!
//! ```text
//! DelayedRelative:  ẋᵢ = Σⱼ aᵢⱼ [xⱼ(t−τ) − xᵢ(t−τ)]      ẋ = −L x(t−τ)
//! CurrentSelf:      ẋᵢ = Σⱼ aᵢⱼ [xⱼ(t−τ) − xᵢ(t)]        ẋ = −D x(t) + A x(t−τ)
//! ```
//!
//! The first is what a relative measurement gives: the whole *difference* arrives over the link, so
//! both terms are as old as the link. The second is what an agent that knows its own state — from
//! its own GNSS receiver, now — can do with the same message. Same graph, same links, same latency.
//!
//! [`Protocol::DelayedRelative`] is the case Olfati-Saber & Murray bound (*Consensus problems in
//! networks of agents with switching topology and time-delays*, IEEE TAC 49(9), 2004): `L` is
//! symmetric, so an orthogonal change of basis decouples the system into `ẏₖ = −λₖ yₖ(t−τ)`, one per
//! Laplacian eigenvalue, and the scalar delay equation loses stability at `λτ = π/2`. The binding
//! mode is the largest eigenvalue:
//!
//! ```text
//! τ_max = π / (2 λ_max(L))
//! ```
//!
//! [`Protocol::CurrentSelf`] has **no such bound**. `derivations/consensus_delay.mac` eliminates the
//! delay from its characteristic equation the same way and gets the crossing condition
//! `ω² = −λ(2d − λ)`, which on a graph — where `0 ≤ λ ≤ 2d` — has no real solution but `ω = 0`, and
//! the origin is the consensus direction itself rather than an instability. The general non-regular
//! case does not decouple, but a crossing at `s = jω` would need `(jωI + D)v = e^{−jωτ} A v`, hence
//! `1 ≤ ‖(jωI + D)⁻¹A‖_∞ = maxᵢ dᵢ/√(ω² + dᵢ²)`, which is strictly below 1 for every `ω ≠ 0`. So
//! consensus survives *any* delay, on any connected graph, at any gain.
//!
//! That is the engineering content of this module, and it is worth stating in the direction that is
//! useful: the delay bound everybody quotes is a property of the **protocol**, not of the link. A
//! formation that feeds its own current state into the comparison is delay-independent; one that
//! differences two equally-old states is not. The measured critical delays are in the tests and in
//! `docs/verification.md` §4.15.
//!
//! # What the state is, and what it is not
//!
//! The state here is one scalar per agent, and the formation problem is three copies of it: the
//! 3-D dynamics are `L ⊗ I₃`, whose spectrum is the same set of eigenvalues with multiplicity three,
//! so the delay bound is unchanged and running three independent copies is not an approximation.
//!
//! Consensus drives every agent to the *same* value, which for absolute positions would be a
//! collision. A formation runs the identical law on the offset error `zᵢ = xᵢ − δᵢ` for a fixed
//! shape `δ`; because the law is linear and `δ` is constant, `ż` obeys exactly the equations above
//! and nothing here needs to know about `δ`. The shape is the caller's, and the collision question
//! it raises is answered where collision questions are answered — `spacelab_estimation::conjunction`
//! — rather than by a separation check invented here.
//!
//! # Layering
//!
//! `λ_max(L)` needs a symmetric eigensolver and there is a tested one two crates down, in
//! [`spacelab_propagation::jacobi_symmetric`]. Depending on a *propagation* crate for linear algebra
//! is defensible here only because this crate already depends on it for the plant and the
//! integrator; a second consumer outside that dependency would be the signal to move the solver to a
//! shared crate rather than to copy it a third time.

use std::vec::Vec;

use spacelab_propagation::jacobi_symmetric;

use crate::delayed::{DelayedSystem, Past};

/// Why a communication graph was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphError {
    /// Fewer than two agents — there is nothing to agree about.
    TooFewAgents,
    /// A weight was not finite, or was negative.
    Weight,
    /// The weight matrix was not symmetric. An undirected link has one gain, not two.
    Asymmetric,
    /// A non-zero diagonal entry: an agent listed as its own neighbour.
    SelfLoop,
    /// The graph is not connected, so at least one pair of agents can never agree.
    Disconnected,
}

/// A fixed undirected communication graph with non-negative link gains.
///
/// The weights are **control gains**, in reciprocal seconds, not adjacency flags: they are what
/// multiplies a position difference to make a velocity command. That is why the eigenvalues below
/// come out in `1/s` and the delay bound in seconds with no further scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Graph<const N: usize> {
    adjacency: [[f64; N]; N],
}

impl<const N: usize> Graph<N> {
    /// A graph from its weighted adjacency matrix.
    ///
    /// # Errors
    ///
    /// [`GraphError`] if there are fewer than two agents, if a weight is negative or not finite, if
    /// the matrix is not exactly symmetric, if any diagonal entry is non-zero, or if the graph is
    /// not connected. Connectivity is checked combinatorially — a breadth-first search over the
    /// non-zero pattern — rather than by testing whether the second eigenvalue is "small enough",
    /// because a weakly connected graph is connected and its `λ₂` is legitimately tiny.
    // Symmetry is checked bit-exactly and not within a tolerance: `laplacian_eigen` is a *symmetric*
    // eigensolver, so an almost-symmetric matrix is not an almost-right input, it is the wrong input.
    #[allow(clippy::float_cmp)]
    pub fn new(adjacency: [[f64; N]; N]) -> Result<Self, GraphError> {
        if N < 2 {
            return Err(GraphError::TooFewAgents);
        }
        for (i, row) in adjacency.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                if !w.is_finite() || w < 0.0 {
                    return Err(GraphError::Weight);
                }
                if i == j && w != 0.0 {
                    return Err(GraphError::SelfLoop);
                }
                if w != adjacency[j][i] {
                    return Err(GraphError::Asymmetric);
                }
            }
        }
        let graph = Self { adjacency };
        if !graph.is_connected() {
            return Err(GraphError::Disconnected);
        }
        Ok(graph)
    }

    /// The weighted adjacency matrix.
    #[must_use]
    pub const fn adjacency(&self) -> &[[f64; N]; N] {
        &self.adjacency
    }

    /// The weighted degrees `dᵢ = Σⱼ aᵢⱼ`.
    #[must_use]
    pub fn degrees(&self) -> [f64; N] {
        let mut d = [0.0; N];
        for (di, row) in d.iter_mut().zip(&self.adjacency) {
            *di = row.iter().sum();
        }
        d
    }

    /// The graph Laplacian `L = D − A`, symmetric positive semi-definite with a null vector `1`.
    #[must_use]
    pub fn laplacian(&self) -> [[f64; N]; N] {
        let mut l = self.adjacency;
        for (i, (row, di)) in l.iter_mut().zip(self.degrees()).enumerate() {
            for w in row.iter_mut() {
                *w = -*w;
            }
            row[i] = di;
        }
        l
    }

    /// The Laplacian eigenvalues, ascending, with their unit eigenvectors as `modes[k]`.
    ///
    /// The eigenvectors are what the disagreement decomposes into: each `modes[k]` evolves as the
    /// scalar delay equation `ẏ = −λₖ y(t−τ)` under [`Protocol::DelayedRelative`], independently of
    /// every other. Measuring the mode rather than the aggregate norm is the difference between
    /// watching the crossing and watching whichever mode happens to dominate the norm.
    #[must_use]
    pub fn laplacian_eigen(&self) -> ([f64; N], [[f64; N]; N]) {
        let l = self.laplacian();
        let rows: Vec<Vec<f64>> = l.iter().map(|row| row.to_vec()).collect();
        let (values, vectors) = jacobi_symmetric(&rows);

        let mut order: [usize; N] = core::array::from_fn(|k| k);
        order.sort_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        let mut sorted = [0.0; N];
        let mut modes = [[0.0; N]; N];
        for (rank, &k) in order.iter().enumerate() {
            sorted[rank] = values[k];
            let column: [f64; N] = core::array::from_fn(|i| vectors[i][k]);
            let norm = libm::sqrt(column.iter().map(|c| c * c).sum::<f64>());
            // Jacobi returns an orthogonal matrix, so the norm is 1 to rounding; dividing anyway
            // costs nothing and means a caller never has to trust that.
            for (m, c) in modes[rank].iter_mut().zip(column) {
                *m = c / norm;
            }
        }
        (sorted, modes)
    }

    /// The Laplacian spectrum, ascending. `λ₁` is zero to rounding; `λ₂` is the algebraic
    /// connectivity and `λ_N` is what sets the delay bound.
    #[must_use]
    pub fn laplacian_spectrum(&self) -> [f64; N] {
        self.laplacian_eigen().0
    }

    /// The algebraic connectivity `λ₂`, the rate at which the slowest disagreement mode decays.
    #[must_use]
    pub fn algebraic_connectivity(&self) -> f64 {
        self.laplacian_spectrum()[1]
    }

    /// `λ_max(L)`, the fastest mode and the one the delay bound is set by.
    #[must_use]
    pub fn largest_laplacian_eigenvalue(&self) -> f64 {
        self.laplacian_spectrum()[N - 1]
    }

    /// Whether every agent can reach every other over links of non-zero gain.
    ///
    /// Combinatorial, by breadth-first search: connectivity is a property of the sparsity pattern
    /// and admits an exact answer, where `λ₂ > tolerance` would need a tolerance nobody can choose
    /// for an arbitrary weighting.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        let mut seen = [false; N];
        let mut stack = std::vec![0usize];
        seen[0] = true;
        let mut count = 1;
        while let Some(i) = stack.pop() {
            for (j, &w) in self.adjacency[i].iter().enumerate() {
                if w > 0.0 && !seen[j] {
                    seen[j] = true;
                    count += 1;
                    stack.push(j);
                }
            }
        }
        count == N
    }
}

/// What an agent does with a message that arrived late.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// `ẋᵢ = Σⱼ aᵢⱼ [xⱼ(t−τ) − xᵢ(t−τ)]` — the relative measurement itself is what is delayed, so
    /// both terms are equally old. This is the protocol Olfati-Saber & Murray bound.
    DelayedRelative,
    /// `ẋᵢ = Σⱼ aᵢⱼ [xⱼ(t−τ) − xᵢ(t)]` — the agent's own state is current and only the neighbour's
    /// is late.
    CurrentSelf,
}

/// The largest delay at which a protocol still reaches consensus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DelayBound {
    /// Consensus is lost above this delay, in seconds.
    Seconds(f64),
    /// No delay destroys consensus. The convergence *rate* still falls with delay; what is
    /// unbounded is the stability, not the performance.
    Unbounded,
}

impl DelayBound {
    /// The bound in seconds, or `f64::INFINITY` when there is none.
    #[must_use]
    pub const fn seconds(self) -> f64 {
        match self {
            Self::Seconds(t) => t,
            Self::Unbounded => f64::INFINITY,
        }
    }
}

/// A consensus law running on a graph, over links that deliver `delay` seconds late.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsensusNetwork<const N: usize> {
    graph: Graph<N>,
    protocol: Protocol,
    delay: [f64; 1],
}

impl<const N: usize> ConsensusNetwork<N> {
    /// `None` unless the delay is finite and strictly positive. A zero delay is refused rather than
    /// treated as a limit: the method of steps needs a positive shortest delay to bound its mesh,
    /// and an undelayed consensus is an ordinary differential equation that belongs to a different
    /// driver.
    #[must_use]
    pub fn new(graph: Graph<N>, protocol: Protocol, delay: f64) -> Option<Self> {
        (delay.is_finite() && delay > 0.0).then_some(Self {
            graph,
            protocol,
            delay: [delay],
        })
    }

    /// The graph the law runs on.
    #[must_use]
    pub const fn graph(&self) -> &Graph<N> {
        &self.graph
    }

    /// The protocol.
    #[must_use]
    pub const fn protocol(&self) -> Protocol {
        self.protocol
    }

    /// The link delay, seconds.
    #[must_use]
    pub const fn delay(&self) -> f64 {
        self.delay[0]
    }

    /// The delay bound for this graph and protocol, derived in `derivations/consensus_delay.mac`.
    ///
    /// Does not depend on [`Self::delay`] — it is what that delay has to stay under, and comparing
    /// the two is [`Self::within_bound`].
    #[must_use]
    pub fn delay_bound(&self) -> DelayBound {
        match self.protocol {
            Protocol::DelayedRelative => DelayBound::Seconds(
                core::f64::consts::PI / (2.0 * self.graph.largest_laplacian_eigenvalue()),
            ),
            Protocol::CurrentSelf => DelayBound::Unbounded,
        }
    }

    /// Whether the configured delay is inside the bound.
    #[must_use]
    pub fn within_bound(&self) -> bool {
        self.delay() < self.delay_bound().seconds()
    }

    /// The disagreement `‖x − x̄1‖`, which is what consensus drives to zero. The mean is subtracted
    /// rather than a designated leader's value, because no agent is a leader and the mean is the
    /// quantity the dynamics conserve.
    #[must_use]
    pub fn disagreement(state: &[f64; N]) -> f64 {
        #[allow(clippy::cast_precision_loss)] // N is an agent count, far inside f64's exact range.
        let mean = state.iter().sum::<f64>() / N as f64;
        libm::sqrt(state.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>())
    }
}

impl<const N: usize> DelayedSystem<N> for ConsensusNetwork<N> {
    fn delays(&self) -> &[f64] {
        &self.delay
    }

    fn derivative(&self, t: f64, y: &[f64; N], past: &impl Past<N>, dydt: &mut [f64; N]) {
        let old = past.at(t - self.delay[0]);
        let own = match self.protocol {
            Protocol::DelayedRelative => &old,
            Protocol::CurrentSelf => y,
        };
        for ((di, row), &xi) in dydt.iter_mut().zip(&self.graph.adjacency).zip(own.iter()) {
            *di = row.iter().zip(&old).map(|(a, xj)| a * (xj - xi)).sum();
        }
    }
}

#[cfg(test)]
mod tests {
    // `excessive_precision`: the Maxima delay bound is quoted verbatim at its 40 printed digits.
    #![allow(clippy::excessive_precision, clippy::expect_used, clippy::float_cmp)]

    use super::*;

    /// The path graph `1 — 2 — 3` with unit gains: degrees `[1, 2, 1]`, and the Laplacian spectrum
    /// is `{0, 1, 3}` in closed form.
    fn path3() -> Graph<3> {
        Graph::new([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).expect("a connected path")
    }

    /// The cycle `C₆` at unit gain, whose Laplacian eigenvalues are `2(1 − cos(2πj/6))`.
    fn cycle6() -> Graph<6> {
        let mut a = [[0.0; 6]; 6];
        for i in 0..6 {
            a[i][(i + 1) % 6] = 1.0;
            a[(i + 1) % 6][i] = 1.0;
        }
        Graph::new(a).expect("a connected cycle")
    }

    #[test]
    fn the_laplacian_spectrum_is_the_closed_form_one() {
        // Path P₃: {0, 1, 3}.
        let got = path3().laplacian_spectrum();
        for (g, w) in got.iter().zip(&[0.0, 1.0, 3.0]) {
            assert!((g - w).abs() < 1e-13, "{g} against {w}");
        }

        // Cycle C₆: 2(1 − cos(2πj/6)) for j = 0..5, i.e. {0, 1, 1, 3, 3, 4}.
        let got = cycle6().laplacian_spectrum();
        let mut want: [f64; 6] = core::array::from_fn(|j| {
            #[allow(clippy::cast_precision_loss)]
            let angle = core::f64::consts::TAU * j as f64 / 6.0;
            2.0 * (1.0 - libm::cos(angle))
        });
        want.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let mut worst = 0.0_f64;
        for (g, w) in got.iter().zip(&want) {
            worst = worst.max((g - w).abs());
        }
        println!("C6 spectrum {got:?}, worst deviation {worst:.3e}");
        assert!(worst < 1e-13, "C₆ spectrum deviates by {worst:.3e}");
    }

    /// `L 1 = 0` exactly, in the sense that the row sums cancel — the conservation law consensus
    /// rests on. Checked on the matrix rather than through the eigensolver, so the two claims about
    /// the null vector are independent.
    #[test]
    fn the_laplacian_annihilates_the_all_ones_vector() {
        for row in cycle6().laplacian() {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-15, "row sum {sum:e}");
        }
    }

    #[test]
    fn connectivity_agrees_with_the_second_eigenvalue() {
        assert!(path3().is_connected());
        assert!(path3().algebraic_connectivity() > 1e-12);

        // Two disjoint edges: connected as a *pattern* only within each pair, so `new` refuses it.
        let split = Graph::new([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]);
        assert_eq!(split, Err(GraphError::Disconnected));
    }

    #[test]
    fn a_malformed_graph_is_refused_rather_than_answered() {
        assert_eq!(
            Graph::new([[0.0, 1.0], [2.0, 0.0]]),
            Err(GraphError::Asymmetric)
        );
        assert_eq!(
            Graph::new([[1.0, 1.0], [1.0, 0.0]]),
            Err(GraphError::SelfLoop)
        );
        assert_eq!(
            Graph::new([[0.0, -1.0], [-1.0, 0.0]]),
            Err(GraphError::Weight)
        );
        assert_eq!(
            Graph::new([[0.0, f64::NAN], [f64::NAN, 0.0]]),
            Err(GraphError::Weight)
        );
        assert_eq!(Graph::new([[0.0f64; 1]; 1]), Err(GraphError::TooFewAgents));
    }

    /// `τ_max λ_max = π/2` for every graph and every weighting, which is the shape of the answer
    /// rather than one instance of it: a formation buys delay margin only by lowering the gain that
    /// produced `λ_max`, and that slows consensus by the same factor.
    #[test]
    fn the_delay_bound_is_pi_over_two_lambda_max() {
        for gain in [1.0e-3, 1.0e-2, 0.5, 7.0] {
            let mut a = [[0.0; 6]; 6];
            for i in 0..6 {
                a[i][(i + 1) % 6] = gain;
                a[(i + 1) % 6][i] = gain;
            }
            let graph = Graph::new(a).expect("a connected cycle");
            let net = ConsensusNetwork::new(graph, Protocol::DelayedRelative, 1.0)
                .expect("a positive delay");
            let product = net.delay_bound().seconds() * graph.largest_laplacian_eigenvalue();
            assert!(
                (product - core::f64::consts::FRAC_PI_2).abs() < 1e-14,
                "τ_max λ_max = {product} at gain {gain}"
            );
        }
        // C₆ at 0.01 /s: λ_max = 4 × 0.01, so τ_max = π/0.08 = 39.2699081698724 s, the value
        // `derivations/consensus_delay.mac` prints at 40 digits.
        let mut a = [[0.0; 6]; 6];
        for i in 0..6 {
            a[i][(i + 1) % 6] = 0.01;
            a[(i + 1) % 6][i] = 0.01;
        }
        let net = ConsensusNetwork::new(
            Graph::new(a).expect("a connected cycle"),
            Protocol::DelayedRelative,
            1.0,
        )
        .expect("a positive delay");
        let cas = 3.926_990_816_987_241_548_078_304_229_099_378_605_247e1;
        let got = net.delay_bound().seconds();
        println!("C6 at 0.01/s: τ_max = {got:.15} s against the CAS {cas:.15} s");
        assert!((got - cas).abs() / cas < 1e-15, "{got} against {cas}");
    }

    #[test]
    fn the_current_self_protocol_has_no_bound() {
        let net = ConsensusNetwork::new(cycle6(), Protocol::CurrentSelf, 1.0e6)
            .expect("a positive delay");
        assert_eq!(net.delay_bound(), DelayBound::Unbounded);
        assert_eq!(net.delay_bound().seconds(), f64::INFINITY);
        assert!(net.within_bound());
    }

    #[test]
    fn a_delay_that_is_not_a_delay_is_refused() {
        assert!(ConsensusNetwork::new(path3(), Protocol::DelayedRelative, 0.0).is_none());
        assert!(ConsensusNetwork::new(path3(), Protocol::DelayedRelative, -1.0).is_none());
        assert!(ConsensusNetwork::new(path3(), Protocol::DelayedRelative, f64::NAN).is_none());
        assert!(ConsensusNetwork::new(path3(), Protocol::DelayedRelative, f64::INFINITY).is_none());
    }

    /// The right-hand side is the matrix product it claims to be. Checked against `−L x` and
    /// `−D x + A x_old` assembled separately, with a history that differs from the current state so
    /// the two protocols cannot coincide by accident.
    #[test]
    fn the_derivative_is_the_matrix_form() {
        struct Fixed([f64; 3]);
        impl Past<3> for Fixed {
            fn at(&self, _s: f64) -> [f64; 3] {
                self.0
            }
        }

        let graph = path3();
        let (now, old) = ([1.0, -2.0, 0.5], [0.25, 3.0, -1.5]);
        let past = Fixed(old);
        let l = graph.laplacian();
        let d = graph.degrees();

        let mut got = [0.0; 3];
        ConsensusNetwork::new(graph, Protocol::DelayedRelative, 1.0)
            .expect("a positive delay")
            .derivative(1.0, &now, &past, &mut got);
        for i in 0..3 {
            let want: f64 = -(0..3).map(|j| l[i][j] * old[j]).sum::<f64>();
            assert!((got[i] - want).abs() < 1e-15, "{} against {want}", got[i]);
        }

        ConsensusNetwork::new(graph, Protocol::CurrentSelf, 1.0)
            .expect("a positive delay")
            .derivative(1.0, &now, &past, &mut got);
        for i in 0..3 {
            let neighbours: f64 = (0..3).map(|j| graph.adjacency()[i][j] * old[j]).sum();
            let want = neighbours - d[i] * now[i];
            assert!((got[i] - want).abs() < 1e-15, "{} against {want}", got[i]);
        }
    }

    /// The mean is conserved by `DelayedRelative` and by `CurrentSelf` alike at the instant the two
    /// states agree, because `1ᵀL = 0`. Away from agreement only the first conserves it — `CurrentSelf`
    /// trades the exact invariant for the unbounded margin, and that trade is worth knowing about.
    #[test]
    fn the_delayed_relative_protocol_conserves_the_mean() {
        struct Fixed([f64; 6]);
        impl Past<6> for Fixed {
            fn at(&self, _s: f64) -> [f64; 6] {
                self.0
            }
        }
        let graph = cycle6();
        let now = [1.0, -2.0, 0.5, 3.0, -1.0, 0.25];
        let past = Fixed([0.1, 0.7, -3.0, 2.0, 0.0, -0.4]);

        let mut rate = [0.0; 6];
        ConsensusNetwork::new(graph, Protocol::DelayedRelative, 1.0)
            .expect("a positive delay")
            .derivative(0.0, &now, &past, &mut rate);
        let drift: f64 = rate.iter().sum();
        assert!(drift.abs() < 1e-15, "the mean drifts at {drift:e}");

        ConsensusNetwork::new(graph, Protocol::CurrentSelf, 1.0)
            .expect("a positive delay")
            .derivative(0.0, &now, &past, &mut rate);
        let drift: f64 = rate.iter().sum();
        println!("CurrentSelf mean drift with a mismatched history: {drift:e}");
        assert!(drift.abs() > 1e-6, "the mean should not be conserved here");
    }

    #[test]
    fn disagreement_is_zero_exactly_at_consensus() {
        assert_eq!(ConsensusNetwork::<4>::disagreement(&[2.5; 4]), 0.0);
        // Two agents at ±1 about their mean: ‖(1, −1)‖ = √2.
        let got = ConsensusNetwork::<2>::disagreement(&[1.0, -1.0]);
        assert!((got - core::f64::consts::SQRT_2).abs() < 1e-15, "{got}");
    }
    /// The eigenvectors are returned normalised and orthogonal — the pair of properties that make
    /// them a change of basis rather than three numbers each. `P₃` is the case that shows it: its
    /// middle mode is `(1, 0, −1)/√2`, and a norm accumulated as `Σ c/c` rather than `Σ c²` is
    /// `NaN` the moment one component is exactly zero.
    #[test]
    fn the_laplacian_modes_are_an_orthonormal_basis() {
        let (values, modes) = path3().laplacian_eigen();
        for (rank, mode) in modes.iter().enumerate() {
            let norm = libm::sqrt(mode.iter().map(|c| c * c).sum::<f64>());
            println!(
                "mode {rank} (lambda = {:.6}): {mode:?}, norm {norm:.17}",
                values[rank]
            );
            assert!((norm - 1.0).abs() < 1e-14, "mode {rank} has norm {norm}");
        }
        for a in 0..3 {
            for b in (a + 1)..3 {
                let inner: f64 = modes[a].iter().zip(modes[b]).map(|(x, y)| x * y).sum();
                assert!(inner.abs() < 1e-14, "modes {a} and {b} overlap by {inner}");
            }
        }
        // The middle mode is the one with the exact zero that a `c / c` norm would trip over.
        assert!(
            modes[1][1].abs() < 1e-15,
            "P3's middle mode is not centred: {:?}",
            modes[1]
        );
    }

    /// `λ₂` is a property of the graph and not a constant. `K₃` has `λ₂ = 3` and `P₃` has `λ₂ = 1`,
    /// so a fixture that used only the path could not tell the value from the unit.
    #[test]
    fn the_algebraic_connectivity_is_the_second_eigenvalue_of_this_graph() {
        let complete = Graph::new([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
            .expect("K3 is connected");
        assert!((complete.algebraic_connectivity() - 3.0).abs() < 1e-14);
        assert!((path3().algebraic_connectivity() - 1.0).abs() < 1e-14);
        assert!(
            (complete.algebraic_connectivity() - complete.laplacian_spectrum()[1]).abs() < 1e-15
        );
    }

    /// The configured delay is carried through unchanged, and it is the same number the delayed
    /// system hands the integrator. Those are two readers of one field, and nothing else in the
    /// module compares them.
    #[test]
    fn the_configured_delay_is_what_both_readers_report() {
        let tau = 0.37;
        let net = ConsensusNetwork::new(cycle6(), Protocol::DelayedRelative, tau)
            .expect("a positive delay");
        assert_eq!(net.delay(), tau);
        assert_eq!(DelayedSystem::delays(&net), &[tau]);
    }

    /// Olfati-Saber and Murray's condition is `τ < π/2λ_max`, strictly — at the bound the loop is
    /// marginally stable, which is not stable. So the fixture is built *at* the bound and has to be
    /// refused, with one just under it accepted to show the comparison is not simply false.
    #[test]
    fn the_delay_bound_is_strict_at_the_bound_itself() {
        let graph = cycle6();
        let bound = ConsensusNetwork::new(graph, Protocol::DelayedRelative, 1.0)
            .expect("a positive delay")
            .delay_bound()
            .seconds();

        let at = ConsensusNetwork::new(graph, Protocol::DelayedRelative, bound)
            .expect("a positive delay");
        println!(
            "bound {bound:.17e}, at the bound within = {}",
            at.within_bound()
        );
        assert!(!at.within_bound(), "the bound was treated as admissible");

        let under = ConsensusNetwork::new(graph, Protocol::DelayedRelative, bound * (1.0 - 1e-12))
            .expect("a positive delay");
        assert!(under.within_bound());
        let over = ConsensusNetwork::new(graph, Protocol::DelayedRelative, bound * 1.5)
            .expect("a positive delay");
        assert!(!over.within_bound());
    }

    /// The disagreement subtracts the mean before squaring, and the algebraically identical
    /// `Σx² − N x̄²` is not an acceptable substitute for it. The two agree in exact arithmetic — that
    /// is the identity — and come apart in `f64` the moment the states share a large offset, which
    /// is the normal case for a formation whose members are all at the same altitude.
    ///
    /// Here the offset is `1e6` and the spread is a third of a metre. The mean-subtracted form sums
    /// three terms of size `0.1` and lands `7.6e-11` from the exact answer; the expanded one sums
    /// three terms of size `6.7e5` that have to cancel to `0.247`, and comes out `4.7e-4` wrong.
    /// The offset is deliberately not larger: at `1e10` the expanded form's sum goes negative and
    /// the square root is `NaN`, which is a louder failure than the silent one worth pinning.
    #[test]
    fn the_disagreement_subtracts_the_mean_before_squaring_it() {
        let spread = [0.0, 0.3, 0.7];
        let offset = 1.0e6;
        let state = [offset + spread[0], offset + spread[1], offset + spread[2]];

        let mean = spread.iter().sum::<f64>() / 3.0;
        let want = libm::sqrt(spread.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>());
        let got = ConsensusNetwork::<3>::disagreement(&state);
        println!("disagreement {got:.17e} against the offset-free {want:.17e}");
        assert!(
            (got - want).abs() < 1e-9 * want,
            "{got:e} against {want:e} — the mean was not subtracted first"
        );
    }
}
