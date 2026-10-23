//! Integration when the right-hand side reads the state at a time that has already passed.
//!
//! # Why this is not a wrapper around the existing integrator
//!
//! [`spacelab_propagation::DynamicalSystem::derivative`] is `(t, y, dydt)` — everything the
//! right-hand side may read is in `y`. A delayed system needs `x(t − τ)`, which is not, and the two
//! ways to force it through that signature are both wrong: widening `y` to carry the buffer makes
//! the state dimension depend on the step size, and hiding a `RefCell` behind `&self` makes the
//! derivative non-reentrant while the integrator calls it once per stage. So [`DelayedSystem`] is a
//! separate trait that takes the history as an argument, and the delayed driver is a loop of its
//! own.
//!
//! What it is *not* is a new numerical method. [`spacelab_propagation::DenseOutput`] already
//! reconstructs the state anywhere inside an accepted step, and method-of-steps integration is
//! exactly that interpolant evaluated at a past time. Every step here is taken by the existing
//! [`Method`] and the existing PI controller.
//!
//! # The method of steps, and the invariant that makes it explicit rather than implicit
//!
//! On an interval no longer than the shortest delay, a delayed system *is* an ordinary one: every
//! `t − τ` the right-hand side asks for has already been computed. The proof is worth stating
//! because the whole design rests on it. Let the current interval be `[a, b]` with `b − a ≤ τ_min`,
//! let the step be `[t, t + h] ⊆ [a, b]`, and let a stage sit at `s = t + c·h` with `c ∈ [0, 1]` —
//! true of every explicit Runge–Kutta method here. Then
//!
//! ```text
//! s − τ  ≤  b − τ_min  ≤  a
//! ```
//!
//! so the lookup lands at or before the interval's *start*, where the solution is already frozen.
//! Not merely before the current step — before the whole interval. That is why
//! [`integrate_delayed_on_mesh`] subdivides any mesh gap longer than `τ_min` instead of trusting the
//! step controller, and why the history can be a plain shared reference with no interior mutability
//! anywhere.
//!
//! A delay shorter than a step would break it, and a delay that vanishes (`τ(t) → 0`) or depends on
//! the state breaks it in a way no subdivision fixes: the equation becomes implicit and needs an
//! iteration this module does not have. Both are refused rather than approximated —
//! [`DelayedSystem::delays`] returns constants, and [`DelayError::Delays`] rejects anything not
//! finite and strictly positive.
//!
//! # Breakpoints: the discontinuity a naive run walks over
//!
//! The initial history `φ` is a boundary condition, and where it disagrees with the solution the
//! problem is not smooth. For `x'(t) = −x(t − τ)` with constant `φ`, `x'` jumps at `t₀` — the
//! history is flat, the solution is not — and that jump *propagates*: `x''` jumps at `t₀ + τ`, `x'''`
//! at `t₀ + 2τ`, and so on. Each delay interval integrates the discontinuity once, so the solution
//! gains exactly one order of smoothness per interval. [`breakpoints`] enumerates those times; for
//! several delays they are the sums `t₀ + Σ kᵢτᵢ`.
//!
//! Placing a mesh point at each of them costs nothing and is not optional. A step that *straddles*
//! the first breakpoint sees a solution that is `C¹` and not `C²` there, so its local error is
//! `O(h³)` however high the method's order — and worse, the dense interpolant it stores is then a
//! cubic fitted across a kink, which the next interval reads back as its history. The measured
//! consequence is in the tests and in `docs/verification.md`: the same RK4 driver on the same
//! problem holds order **4.00** on a mesh that lands on the breakpoints and falls to **1.91** on one
//! that does not. Two orders, and nothing complains — a fixed-step run has no error estimate to
//! notice, and an adaptive one is looking at a step it took, not at a kink it stepped over.
//!
//! # What commensurability does and does not buy
//!
//! A delay that is a whole number of steps makes the *step-end* lookups exact: `t − τ` is a node and
//! the interpolant returns the stored sample. It does not make the run interpolation-free, because
//! an explicit Runge–Kutta method also evaluates at interior stages, and `t + c·h − τ` for
//! `0 < c < 1` falls between two nodes whatever `τ/h` is. Only a loop that evaluates once per step —
//! a digital controller with a zero-order hold, which is what [`crate::AttitudeLoop`] runs — reads a
//! history that is entirely stored samples. The test
//! `commensurability_leaves_the_interior_stages_interpolated` records this, because the opposite is
//! the natural assumption and it is wrong.

use std::vec::Vec;

use spacelab_propagation::{
    DenseOutput, DynamicalSystem,
    driver::{
        Controls, HermiteDense, IntegrationError, Method, Trajectory, integrate,
        integrate_fixed_hermite,
    },
    rk4::rk4_step,
};

/// The state at a time the integration has already passed, or before it began.
///
/// For `s` before the initial time this is the problem's initial history `φ(s)`; after it, the
/// computed solution. The two are the same object because the right-hand side cannot tell them
/// apart and should not have to.
pub trait Past<const N: usize> {
    /// The state at time `s`.
    fn at(&self, s: f64) -> [f64; N];
}

/// The constant initial history `φ(s) = state`, which is what a system started from rest is in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantHistory<const N: usize>(pub [f64; N]);

impl<const N: usize> Past<N> for ConstantHistory<N> {
    fn at(&self, _s: f64) -> [f64; N] {
        self.0
    }
}

/// A right-hand side that reads the state at one or more earlier times.
///
/// The delays are constants. State-dependent and vanishing delays are a different problem — they
/// make the equation implicit within a step — and are outside this module rather than silently
/// approximated by it.
pub trait DelayedSystem<const N: usize> {
    /// The delays, each strictly positive and finite. The shortest one bounds the mesh.
    fn delays(&self) -> &[f64];

    /// Evaluate `dy/dt` at `(t, y)`, reading earlier states through `past`.
    fn derivative(&self, t: f64, y: &[f64; N], past: &impl Past<N>, dydt: &mut [f64; N]);
}

/// The initial condition of a delayed system: the history before `t0`, and the state at it.
///
/// Both, and not one derived from the other, because they are allowed to disagree. `y0` is where the
/// vehicle is at `t0`; `history` is where it was before anyone was integrating. Where the two do not
/// meet, `dy/dt` jumps at `t0` and that jump is the first breakpoint — the whole reason
/// [`breakpoints`] exists. Deriving `y0` from `history` would make the commonest case in control
/// (a loop switched on while the plant is already moving) inexpressible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialValue<const N: usize, H> {
    /// `φ(s)` for `s ≤ t0`.
    pub history: H,
    /// The initial time.
    pub t0: f64,
    /// The state at `t0`.
    pub y0: [f64; N],
}

impl<const N: usize, H> InitialValue<N, H> {
    /// The initial condition `φ = history`, starting at `(t0, y0)`.
    pub const fn new(history: H, t0: f64, y0: [f64; N]) -> Self {
        Self { history, t0, y0 }
    }
}

impl<const N: usize> InitialValue<N, ConstantHistory<N>> {
    /// The equilibrium start: a constant history equal to the initial state, so nothing jumps at
    /// `t0` and the first breakpoint is one order smoother than it would otherwise be.
    #[must_use]
    pub const fn at_rest(t0: f64, y0: [f64; N]) -> Self {
        Self {
            history: ConstantHistory(y0),
            t0,
            y0,
        }
    }
}

/// Why a delayed integration could not run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DelayError {
    /// The delay list was empty, or a delay was not finite and strictly positive.
    Delays,
    /// The mesh had fewer than two points, was not strictly increasing, or was not finite.
    Mesh,
    /// Subdividing the mesh to the shortest delay would need more intervals than [`MAX_INTERVALS`].
    TooManyIntervals {
        /// How many were asked for.
        asked: f64,
    },
    /// A step count of zero was asked of the fixed-step driver.
    NoSteps,
    /// The underlying step controller failed.
    Integration(IntegrationError),
}

impl From<IntegrationError> for DelayError {
    fn from(e: IntegrationError) -> Self {
        Self::Integration(e)
    }
}

/// The most intervals the mesh may be subdivided into.
///
/// A span of a year at a one-second delay is 3.15e7 intervals, which is past any use this driver
/// has; the cap is a ceiling on a runaway argument, not a budget. It is checked before the loop is
/// entered rather than counted inside it, because the count is known from the span and the delay.
pub const MAX_INTERVALS: usize = 1_000_000;

/// How many levels of propagated discontinuity [`integrate_delayed`] tracks.
///
/// The solution gains one order of smoothness per delay interval, so past level `p` it is `Cᵖ` and a
/// method of order `p` is no longer limited by the breakpoint. Eight is the highest order in the
/// workspace (DOP853), so tracking eight levels covers every method here; tracking more would place
/// mesh points where nothing happens.
pub const BREAKPOINT_LEVELS: usize = 8;

/// The times at which the propagated discontinuity of a delayed system lands, strictly inside
/// `(t0, t1)` and sorted.
///
/// These are `t0 + Σ kᵢτᵢ` for every multi-index with `1 ≤ Σ kᵢ ≤ levels`. Times closer together
/// than eight ulps of the span are merged: with several delays, different multi-indices can name the
/// same instant (`τ = [1, ½]` reaches `t0 + 1` two ways) and a mesh does not want it twice.
///
/// Returns `None` if the delays are not all finite and strictly positive, or if the set would exceed
/// [`MAX_INTERVALS`].
#[must_use]
pub fn breakpoints(delays: &[f64], t0: f64, t1: f64, levels: usize) -> Option<Vec<f64>> {
    if smallest_delay(delays).is_err() || !t0.is_finite() || !t1.is_finite() || t1 <= t0 {
        return None;
    }
    let span = t1 - t0;
    let tol = 8.0 * f64::EPSILON * span;
    let mut frontier = std::vec![0.0f64];
    let mut all: Vec<f64> = Vec::new();
    for _ in 0..levels {
        let mut next: Vec<f64> = Vec::new();
        for &s in &frontier {
            for &tau in delays {
                let v = s + tau;
                if v < span {
                    next.push(v);
                }
            }
        }
        dedup_sorted(&mut next, tol);
        if all.len() + next.len() > MAX_INTERVALS {
            return None;
        }
        all.extend_from_slice(&next);
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    dedup_sorted(&mut all, tol);
    Some(all.into_iter().map(|v| t0 + v).collect())
}

/// Sort and collapse values within `tol` of each other, keeping the first of each run.
fn dedup_sorted(v: &mut Vec<f64>, tol: f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    v.dedup_by(|a, b| (*a - *b).abs() <= tol);
}

/// The shortest delay, or [`DelayError::Delays`] if any is not finite and strictly positive.
fn smallest_delay(delays: &[f64]) -> Result<f64, DelayError> {
    if delays.is_empty() || !delays.iter().all(|d| d.is_finite() && *d > 0.0) {
        return Err(DelayError::Delays);
    }
    Ok(delays.iter().copied().fold(f64::INFINITY, f64::min))
}

/// The solution of a delayed system: the initial history and every arc computed after it, evaluable
/// at any time on either side of `t0`.
///
/// Held as one arc per mesh interval rather than one flat list of segments, because that is what
/// makes the delayed lookup select the right interpolant: [`spacelab_propagation::DenseOutput`] is
/// only valid inside the step it came from, and an arc boundary is exactly where the solution stops
/// being smooth.
#[derive(Debug, Clone)]
pub struct DelayedSolution<const N: usize, D, H> {
    history: H,
    t0: f64,
    arcs: Vec<Trajectory<N, D>>,
    ends: Vec<f64>,
    y_final: [f64; N],
    accepted: u32,
    rejected: u32,
}

impl<const N: usize, D: DenseOutput<N>, H: Past<N>> DelayedSolution<N, D, H> {
    /// A solution holding only its initial condition — what the first interval reads as its history.
    fn opening(initial: InitialValue<N, H>) -> Self {
        Self {
            history: initial.history,
            t0: initial.t0,
            arcs: Vec::new(),
            ends: Vec::new(),
            y_final: initial.y0,
            accepted: 0,
            rejected: 0,
        }
    }

    /// Append a completed arc ending at `end`.
    fn push(&mut self, arc: Trajectory<N, D>, end: f64) {
        let (accepted, rejected) = arc.step_counts();
        self.accepted += accepted;
        self.rejected += rejected;
        self.arcs.push(arc);
        self.ends.push(end);
    }

    /// The state at `t`: the initial history at or before `t0`, the computed solution after it.
    ///
    /// Past the final time this extrapolates the last arc, the same contract
    /// [`Trajectory::eval`] carries.
    #[must_use]
    pub fn eval(&self, t: f64) -> [f64; N] {
        // `arcs` is empty only while the first interval is being integrated, and the invariant in
        // the module documentation puts every lookup at or before `t0` there, so this is the
        // initial history rather than a fallback for a case that can happen later.
        if self.arcs.is_empty() || t <= self.t0 {
            return self.history.at(t);
        }
        let k = match self
            .ends
            .binary_search_by(|e| e.partial_cmp(&t).unwrap_or(core::cmp::Ordering::Less))
        {
            Ok(i) | Err(i) => i.min(self.arcs.len() - 1),
        };
        self.arcs[k].eval(t)
    }

    /// The state at the end of the integration.
    #[must_use]
    pub fn final_state(&self) -> [f64; N] {
        self.y_final
    }

    /// Accepted and rejected step counts, summed over every arc.
    #[must_use]
    pub fn step_counts(&self) -> (u32, u32) {
        (self.accepted, self.rejected)
    }

    /// The mesh the solution was built on: the interval boundaries, ascending.
    #[must_use]
    pub fn arc_ends(&self) -> &[f64] {
        &self.ends
    }
}

impl<const N: usize, D: DenseOutput<N>, H: Past<N>> Past<N> for DelayedSolution<N, D, H> {
    fn at(&self, s: f64) -> [f64; N] {
        self.eval(s)
    }
}

/// A delayed system with its history pinned, so it satisfies the ordinary
/// [`DynamicalSystem`] contract for the length of one interval.
///
/// This is the whole adapter. It holds no buffer and copies no state: the interval invariant means
/// the history it points at is complete for every lookup the interval can make.
struct Frozen<'a, S, P> {
    sys: &'a S,
    past: &'a P,
}

impl<const N: usize, S: DelayedSystem<N>, P: Past<N>> DynamicalSystem<N> for Frozen<'_, S, P> {
    fn derivative(&self, t: f64, y: &[f64; N], dydt: &mut [f64; N]) {
        self.sys.derivative(t, y, self.past, dydt);
    }
}

/// Validate a mesh and return the number of equal pieces each of its gaps must be cut into so that
/// no piece is longer than `tau_min`.
///
/// `arc_ends` are the boundaries strictly after `t0`, ascending, the last being the final time.
fn mesh_pieces(t0: f64, arc_ends: &[f64], tau_min: f64) -> Result<Vec<usize>, DelayError> {
    if arc_ends.is_empty() || !t0.is_finite() || !arc_ends.iter().all(|t| t.is_finite()) {
        return Err(DelayError::Mesh);
    }
    let mut pieces = Vec::with_capacity(arc_ends.len());
    let mut total = 0.0f64;
    let mut start = t0;
    for &end in arc_ends {
        if end <= start {
            return Err(DelayError::Mesh);
        }
        let asked = ((end - start) / tau_min).ceil().max(1.0);
        start = end;
        total += asked;
        // `asked` is `ceil` of a finite non-negative quotient, so `total` is finite or `+inf`;
        // never NaN, which is why a plain `>` is the whole test.
        #[allow(clippy::cast_precision_loss)]
        if total > MAX_INTERVALS as f64 {
            return Err(DelayError::TooManyIntervals { asked: total });
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        pieces.push(asked as usize);
    }
    Ok(pieces)
}

/// The `index`-th of `count` equal pieces of `[start, end]`, with the last ending exactly at `end`.
#[allow(clippy::cast_precision_loss)]
fn piece(start: f64, end: f64, index: usize, count: usize) -> (f64, f64) {
    let step = (end - start) / count as f64;
    let from = start + index as f64 * step;
    // Exactly `end` rather than `start + count * step`, so the arcs join where the mesh says and
    // the next interval's history lookup does not miss the join by an ulp.
    let to = if index + 1 == count {
        end
    } else {
        start + (index + 1) as f64 * step
    };
    (from, to)
}

/// Integrate a delayed system on a caller-supplied mesh, adapting the step inside each arc.
///
/// `arc_ends` are the arc boundaries *after* `initial.t0`, strictly increasing, the last being the
/// final time — the initial time is not repeated there, so a mesh cannot disagree with the initial
/// condition about where the integration starts. Any gap longer than the shortest delay is cut into
/// equal pieces that are not: required for correctness, not tuning, by the invariant in the module
/// documentation.
///
/// Use this when the breakpoints are known for a reason [`breakpoints`] cannot see: a commanded
/// attitude change at a scheduled time, a thruster firing, a mode switch. Otherwise use
/// [`integrate_delayed`], which builds the mesh from the delays.
///
/// # Errors
///
/// [`DelayError::Delays`] for a delay that is not finite and strictly positive,
/// [`DelayError::Mesh`] for a mesh that is not strictly increasing and finite,
/// [`DelayError::TooManyIntervals`] if the subdivision exceeds [`MAX_INTERVALS`], and
/// [`DelayError::Integration`] for whatever the step controller reports.
pub fn integrate_delayed_on_mesh<const N: usize, M, S, H>(
    method: &M,
    sys: &S,
    initial: InitialValue<N, H>,
    arc_ends: &[f64],
    h0: f64,
    controls: Controls,
) -> Result<DelayedSolution<N, M::Dense, H>, DelayError>
where
    M: Method<N>,
    S: DelayedSystem<N>,
    H: Past<N>,
{
    let tau_min = smallest_delay(sys.delays())?;
    let pieces = mesh_pieces(initial.t0, arc_ends, tau_min)?;

    let mut sol = DelayedSolution::opening(initial);
    let mut y = sol.y_final;
    let mut start = sol.t0;
    for (&end, &count) in arc_ends.iter().zip(&pieces) {
        for index in 0..count {
            let (from, to) = piece(start, end, index, count);
            let arc = {
                let frozen = Frozen {
                    sys,
                    past: &sol as &DelayedSolution<N, M::Dense, H>,
                };
                integrate(method, &frozen, from, &y, to, h0, controls)?
            };
            y = arc.final_state();
            sol.push(arc, to);
        }
        start = end;
    }
    sol.y_final = y;
    Ok(sol)
}

/// Integrate a delayed system to `t1`, placing an arc boundary at every propagated discontinuity.
///
/// The mesh is the [`breakpoints`] to [`BREAKPOINT_LEVELS`] and then `t1`. That is the whole
/// difference between this and an ordinary adaptive run with a history buffer bolted on, and it is
/// the difference between holding the method's order and losing it.
///
/// # Errors
///
/// As [`integrate_delayed_on_mesh`], plus [`DelayError::Mesh`] if `t1` is not after `t0`.
pub fn integrate_delayed<const N: usize, M, S, H>(
    method: &M,
    sys: &S,
    initial: InitialValue<N, H>,
    t1: f64,
    h0: f64,
    controls: Controls,
) -> Result<DelayedSolution<N, M::Dense, H>, DelayError>
where
    M: Method<N>,
    S: DelayedSystem<N>,
    H: Past<N>,
{
    let mesh = breakpoint_mesh(sys.delays(), initial.t0, t1)?;
    integrate_delayed_on_mesh(method, sys, initial, &mesh, h0, controls)
}

/// Every propagated breakpoint strictly inside `(t0, t1)`, then `t1`.
fn breakpoint_mesh(delays: &[f64], t0: f64, t1: f64) -> Result<Vec<f64>, DelayError> {
    smallest_delay(delays)?;
    let inner = breakpoints(delays, t0, t1, BREAKPOINT_LEVELS).ok_or(DelayError::Mesh)?;
    let mut mesh: Vec<f64> = inner.into_iter().filter(|t| *t > t0 && *t < t1).collect();
    mesh.push(t1);
    Ok(mesh)
}

/// Integrate a delayed system on a caller-supplied mesh with a fixed number of equal RK4 steps per
/// piece, and cubic-Hermite dense output.
///
/// The adaptive driver is the one to reach for; this exists because a convergence order is a
/// statement about `h`, and an adaptive run has no single `h` to make it about. It is also the shape
/// a digital control loop runs in — a fixed rate, chosen once — so the two uses coincide.
///
/// # Errors
///
/// As [`integrate_delayed_on_mesh`], plus [`DelayError::NoSteps`] for `steps_per_piece == 0`.
pub fn integrate_delayed_fixed_on_mesh<const N: usize, S, H>(
    sys: &S,
    initial: InitialValue<N, H>,
    arc_ends: &[f64],
    steps_per_piece: usize,
) -> Result<DelayedSolution<N, HermiteDense<N>, H>, DelayError>
where
    S: DelayedSystem<N>,
    H: Past<N>,
{
    if steps_per_piece == 0 {
        return Err(DelayError::NoSteps);
    }
    let tau_min = smallest_delay(sys.delays())?;
    let pieces = mesh_pieces(initial.t0, arc_ends, tau_min)?;

    let mut sol = DelayedSolution::opening(initial);
    let mut y = sol.y_final;
    let mut start = sol.t0;
    for (&end, &count) in arc_ends.iter().zip(&pieces) {
        for index in 0..count {
            let (from, to) = piece(start, end, index, count);
            let arc = {
                let frozen = Frozen {
                    sys,
                    past: &sol as &DelayedSolution<N, HermiteDense<N>, H>,
                };
                integrate_fixed_hermite(
                    from,
                    to,
                    &y,
                    steps_per_piece,
                    |t, y| {
                        let mut dydt = [0.0; N];
                        frozen.derivative(t, y, &mut dydt);
                        dydt
                    },
                    |t, y, h| Ok(rk4_step(&frozen, t, y, h)),
                )?
            };
            y = arc.final_state();
            sol.push(arc, to);
        }
        start = end;
    }
    sol.y_final = y;
    Ok(sol)
}

#[cfg(test)]
mod tests {
    // The reference values are Maxima's 40-digit output verbatim, per the workspace convention for
    // constants taken from a primary source: truncating them here would make the fixture's
    // provenance unverifiable against the file that produced it.
    #![allow(clippy::excessive_precision, clippy::expect_used, clippy::float_cmp)]

    use spacelab_propagation::{Dop853, Tolerance, driver::Controls};

    use super::*;

    /// `x'(t) = −x(t − 1)` with the constant history `x ≡ 1` for `t ≤ 0` — the textbook delayed
    /// equation, and the one whose method-of-steps solution is a closed-form piecewise polynomial.
    struct Linear {
        tau: [f64; 1],
    }

    impl Linear {
        const fn new() -> Self {
            Self { tau: [1.0] }
        }
    }

    impl DelayedSystem<1> for Linear {
        fn delays(&self) -> &[f64] {
            &self.tau
        }

        fn derivative(&self, t: f64, _y: &[f64; 1], past: &impl Past<1>, dydt: &mut [f64; 1]) {
            dydt[0] = -past.at(t - self.tau[0])[0];
        }
    }

    /// The exact solution at the integers and half-integers, from `derivations/dde_method_of_steps.mac`.
    ///
    /// Maxima builds the piecewise polynomials by exact rational integration, then checks its own
    /// recursion three ways before printing: `x` and `x'` continuous at every breakpoint, the `x''`
    /// jump exactly 1 at `t = 1` and exactly 0 at `t = 2` (one interval of smoothing), and the DDE
    /// residual `x'(t) + x(t − 1)` identically zero on all eight intervals. The values are 40
    /// significant digits truncated to `f64`.
    const REFERENCE: [(f64, f64); 16] = [
        (0.5, 5.0e-1),
        (1.0, 0.0),
        (1.5, -3.75e-1),
        (2.0, -5.0e-1),
        (2.5, -3.958_333_333_333_333_333e-1),
        (3.0, -1.666_666_666_666_666_667e-1),
        (3.5, 6.510_416_666_666_666_667e-2),
        (4.0, 2.083_333_333_333_333_333e-1),
        (4.5, 2.315_104_166_666_666_667e-1),
        (5.0, 1.583_333_333_333_333_333e-1),
        (5.5, 4.351_128_472_222_222_222e-2),
        (6.0, -5.694_444_444_444_444_444e-2),
        (6.5, -1.078_791_542_658_730_159e-1),
        (7.0, -1.029_761_904_761_904_762e-1),
        (7.5, -5.935_785_202_752_976_190e-2),
        (8.0, -4.538_690_476_190_476_190e-3),
    ];

    fn dop853_controls() -> Controls {
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-12)
    }

    /// The driver against the closed form, at the breakpoints and between them.
    ///
    /// Both matter and for different reasons: the integer points are where the arcs join, and the
    /// half-integers are where the dense interpolant is doing the work — which is also where every
    /// delayed lookup lands, since `t − 1` is a node only by coincidence on an adaptive mesh.
    #[test]
    fn the_adaptive_driver_reproduces_the_closed_form_at_and_between_the_breakpoints() {
        let sys = Linear::new();
        let sol = integrate_delayed(
            &Dop853,
            &sys,
            InitialValue::at_rest(0.0, [1.0]),
            8.0,
            0.1,
            dop853_controls(),
        )
        .expect("the linear delayed equation integrates");

        let mut worst_node: f64 = 0.0;
        let mut worst_interior: f64 = 0.0;
        for (t, exact) in REFERENCE {
            let err = (sol.eval(t)[0] - exact).abs();
            if t.fract() == 0.0 {
                worst_node = worst_node.max(err);
            } else {
                worst_interior = worst_interior.max(err);
            }
        }
        println!(
            "closed form: worst at a breakpoint {worst_node:.3e}, between {worst_interior:.3e}"
        );
        assert!(worst_node < 1e-12, "worst at a breakpoint {worst_node:.3e}");
        assert!(worst_interior < 1e-12, "worst between {worst_interior:.3e}");
        assert!((sol.final_state()[0] - REFERENCE[15].1).abs() < 1e-12);
    }

    /// The headline of this ticket, measured rather than argued.
    ///
    /// The same RK4 driver, the same problem, the same work — the only difference is whether the
    /// mesh lands on the propagated discontinuities. Seven step sizes, each half the last, and the
    /// observed order is the least-squares slope of `log ε` against `log h` over all seven.
    ///
    /// The slope rather than the last ratio, and the reason is in the numbers. The tracked run's
    /// successive ratios are 4.00 six times over and either statistic would do. The untracked run's
    /// are 1.85, 2.93, 1.60, 0.34, 3.32, 1.68 — because where the kink falls *inside* a step changes
    /// with `h`, so the error constant oscillates even though the trend is clean. Quoting the last
    /// ratio of that sequence would publish whichever of 0.34 and 3.32 the loop happened to stop on.
    ///
    /// The untracked mesh starts with a deliberate `0.3` so that neither its first piece nor the
    /// equal pieces after it can coincide with an integer, where every breakpoint of this problem
    /// is. Nothing else about it is unusual: the pieces are still shorter than the delay, so the run
    /// is a legitimate method-of-steps integration that simply does not know where the kinks are.
    #[test]
    fn tracking_the_breakpoints_holds_the_order_and_missing_them_loses_it() {
        let sys = Linear::new();
        let tracked_mesh = breakpoint_mesh(sys.delays(), 0.0, 8.0).expect("mesh");
        let untracked_mesh = [0.3, 8.0];
        let exact = REFERENCE[15].1;

        let run = |mesh: &[f64], steps: usize| -> f64 {
            let sol = integrate_delayed_fixed_on_mesh(
                &sys,
                InitialValue::at_rest(0.0, [1.0]),
                mesh,
                steps,
            )
            .expect("fixed-step run");
            (sol.final_state()[0] - exact).abs()
        };

        let order = |mesh: &[f64]| -> f64 {
            let errs: Vec<f64> = (0..7).map(|k| run(mesh, 4usize << k)).collect();
            let ratios: Vec<String> = errs
                .windows(2)
                .map(|w| std::format!("{:.2}", (w[0] / w[1]).log2()))
                .collect();
            let shown: Vec<String> = errs.iter().map(|e| std::format!("{e:.3e}")).collect();
            // Least squares on (k, log2 ε): halving the step advances k by one, so minus the slope
            // is the order.
            #[allow(clippy::cast_precision_loss)]
            let n = errs.len() as f64;
            #[allow(clippy::cast_precision_loss)]
            let k_bar = (errs.len() - 1) as f64 / 2.0;
            let l_bar = errs.iter().map(|e| e.log2()).sum::<f64>() / n;
            #[allow(clippy::cast_precision_loss)]
            let (num, den) = errs.iter().enumerate().fold((0.0, 0.0), |(a, b), (i, e)| {
                let dk = i as f64 - k_bar;
                (a + dk * (e.log2() - l_bar), b + dk * dk)
            });
            let slope = -num / den;
            println!(
                "mesh {:?}...: errors {shown:?}\n  ratios {ratios:?} -> fitted order {slope:.2}",
                &mesh[..2.min(mesh.len())]
            );
            slope
        };

        let tracked = order(&tracked_mesh);
        let untracked = order(&untracked_mesh);
        println!("observed order: tracked {tracked:.2}, untracked {untracked:.2}");
        assert!(
            tracked > 3.95,
            "RK4 on a breakpoint-aligned mesh must hold order 4, got {tracked:.2}"
        );
        assert!(
            untracked < 3.0,
            "the premise of this ticket is that a straddled breakpoint costs order; \
             the untracked run scored {untracked:.2}, which refutes it"
        );
    }

    /// What the dense output costs, priced rather than assumed negligible.
    ///
    /// Every delayed lookup on an adaptive mesh lands *inside* a step — `t − τ` is a node only by
    /// coincidence — so the interpolant is on the critical path of the answer and not merely of the
    /// output. The comparison is the error where an arc ends, which is an accepted step's own state,
    /// against the error halfway along an arc, which is entirely the interpolant's.
    #[test]
    fn the_dense_interpolant_is_not_the_limiting_error() {
        let sys = Linear::new();
        let sol = integrate_delayed(
            &Dop853,
            &sys,
            InitialValue::at_rest(0.0, [1.0]),
            8.0,
            0.1,
            dop853_controls(),
        )
        .expect("integrates");
        assert_eq!(
            sol.arc_ends(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "one arc per delay interval, so t = 7.0 is a join and t = 7.5 is mid-arc"
        );

        let at_join = (sol.eval(7.0)[0] - REFERENCE[13].1).abs();
        let mid_arc = (sol.eval(7.5)[0] - REFERENCE[14].1).abs();
        println!("dense output: at the join {at_join:.3e}, mid-arc {mid_arc:.3e}");
        assert!(
            mid_arc < 1e-12,
            "mid-arc error {mid_arc:.3e} is the interpolant's, and it is on the critical path"
        );
        assert!(
            mid_arc < 10.0 * at_join.max(f64::EPSILON),
            "the interpolant would be the limiting term: mid-arc {mid_arc:.3e} against join \
             {at_join:.3e}"
        );
    }

    /// A commensurate delay is not an interpolation-free one, and the difference matters because the
    /// opposite is what everyone assumes.
    ///
    /// With `τ = 1` and four steps of `h = ¼`, RK4 evaluates at `t`, `t + h/2` twice and `t + h`. The
    /// step-end stages ask the history for a node and get a stored sample; the two half-step stages
    /// ask for a point between two nodes and get the interpolant, and no choice of `τ/h` changes
    /// that. Exactness of the delayed lookup is a property of a one-evaluation-per-step loop — a
    /// digital controller holding its command across the step — not of the delay dividing the step.
    #[test]
    fn commensurability_leaves_the_interior_stages_interpolated() {
        use std::cell::RefCell;

        struct Recorder {
            tau: [f64; 1],
            asked: RefCell<Vec<f64>>,
        }
        impl DelayedSystem<1> for Recorder {
            fn delays(&self) -> &[f64] {
                &self.tau
            }
            fn derivative(&self, t: f64, _y: &[f64; 1], past: &impl Past<1>, dydt: &mut [f64; 1]) {
                let s = t - self.tau[0];
                self.asked.borrow_mut().push(s);
                dydt[0] = -past.at(s)[0];
            }
        }

        let h = 0.25;
        let sys = Recorder {
            tau: [1.0],
            asked: RefCell::new(Vec::new()),
        };
        integrate_delayed_fixed_on_mesh(&sys, InitialValue::at_rest(0.0, [1.0]), &[1.0], 4)
            .expect("integrates");

        let asked = sys.asked.borrow();
        let off_node = asked
            .iter()
            .filter(|s| (*s / h - (*s / h).round()).abs() > 1e-12)
            .count();
        println!(
            "commensurate tau/h = {:.0}: {} of {} delayed lookups are between nodes",
            1.0 / h,
            off_node,
            asked.len()
        );
        assert!(
            off_node > 0,
            "if every lookup were a node the dense output would be dead weight, and it is not"
        );
    }

    #[test]
    fn breakpoints_are_the_sums_of_the_delays_and_are_not_repeated() {
        let single = breakpoints(&[1.0], 0.0, 4.5, 8).expect("valid");
        assert_eq!(single, std::vec![1.0, 2.0, 3.0, 4.0]);

        // τ = [1, ½] reaches 1 as 1·1 and as 2·½; the mesh wants it once.
        let pair = breakpoints(&[1.0, 0.5], 0.0, 2.1, 8).expect("valid");
        assert_eq!(pair, std::vec![0.5, 1.0, 1.5, 2.0]);

        let offset = breakpoints(&[0.75], 10.0, 12.0, 8).expect("valid");
        assert_eq!(offset, std::vec![10.75, 11.5]);

        assert!(breakpoints(&[], 0.0, 1.0, 8).is_none());
        assert!(breakpoints(&[0.0], 0.0, 1.0, 8).is_none());
        assert!(breakpoints(&[-1.0], 0.0, 1.0, 8).is_none());
        assert!(breakpoints(&[f64::NAN], 0.0, 1.0, 8).is_none());
        assert!(breakpoints(&[1.0], 1.0, 1.0, 8).is_none());
    }

    /// The subdivision is a correctness requirement, not tuning: a mesh gap longer than the shortest
    /// delay would let a stage read a state the current interval has not produced yet.
    #[test]
    fn a_mesh_gap_longer_than_the_delay_is_cut_until_it_is_not() {
        assert_eq!(mesh_pieces(0.0, &[1.0], 1.0), Ok(std::vec![1]));
        assert_eq!(mesh_pieces(0.0, &[2.5], 1.0), Ok(std::vec![3]));
        assert_eq!(mesh_pieces(0.0, &[1.0, 4.0], 1.0), Ok(std::vec![1, 3]));

        assert_eq!(mesh_pieces(0.0, &[], 1.0), Err(DelayError::Mesh));
        assert_eq!(mesh_pieces(1.0, &[0.0], 1.0), Err(DelayError::Mesh));
        assert_eq!(mesh_pieces(0.0, &[0.0], 1.0), Err(DelayError::Mesh));
        assert_eq!(mesh_pieces(0.0, &[2.0, 1.0], 1.0), Err(DelayError::Mesh));
        assert_eq!(mesh_pieces(0.0, &[f64::NAN], 1.0), Err(DelayError::Mesh));
        assert_eq!(mesh_pieces(f64::NAN, &[1.0], 1.0), Err(DelayError::Mesh));
        assert!(matches!(
            mesh_pieces(0.0, &[1e12], 1.0),
            Err(DelayError::TooManyIntervals { .. })
        ));
    }

    #[test]
    fn a_delay_that_is_not_finite_and_positive_is_refused_rather_than_clamped() {
        struct Bad([f64; 1]);
        impl DelayedSystem<1> for Bad {
            fn delays(&self) -> &[f64] {
                &self.0
            }
            fn derivative(&self, _t: f64, _y: &[f64; 1], _p: &impl Past<1>, d: &mut [f64; 1]) {
                d[0] = 0.0;
            }
        }
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = integrate_delayed_fixed_on_mesh(
                &Bad([bad]),
                InitialValue::at_rest(0.0, [1.0]),
                &[1.0],
                4,
            )
            .expect_err("a delay that is not finite and positive has no method of steps");
            assert_eq!(err, DelayError::Delays);
        }
    }

    #[test]
    fn the_history_is_read_before_the_initial_time_and_the_solution_after_it() {
        let sys = Linear::new();
        let sol =
            integrate_delayed_fixed_on_mesh(&sys, InitialValue::at_rest(0.0, [1.0]), &[1.0], 16)
                .expect("integrates");
        assert_eq!(sol.eval(-5.0), [1.0], "before t0 the initial history");
        assert_eq!(sol.eval(0.0), [1.0], "at t0 the initial history");
        // On [0, 1] the exact solution is 1 − t, and RK4 on a constant right-hand side is exact.
        assert!((sol.eval(0.5)[0] - 0.5).abs() < 1e-15);
        assert!((sol.final_state()[0] - 0.0).abs() < 1e-15);
    }

    #[test]
    fn a_zero_step_count_is_refused() {
        assert_eq!(
            integrate_delayed_fixed_on_mesh(
                &Linear::new(),
                InitialValue::at_rest(0.0, [1.0]),
                &[1.0],
                0
            )
            .map(|_| ()),
            Err(DelayError::NoSteps)
        );
    }
    /// The constant history is the whole initial condition of a system started from rest, so what
    /// it returns has to be the state it was built from and not a state of its own choosing.
    #[test]
    fn the_constant_history_returns_the_state_it_was_built_from() {
        let phi = ConstantHistory([2.0, -3.5]);
        assert_eq!(phi.at(-1.0), [2.0, -3.5]);
        assert_eq!(phi.at(0.0), [2.0, -3.5]);
        assert_eq!(phi.at(f64::NEG_INFINITY), [2.0, -3.5]);
    }

    /// The step counts are summed over arcs, and both counters start at zero — so a test that only
    /// looks at the accepted count cannot see the rejected one being accumulated at all.
    ///
    /// Rejections are forced rather than hoped for: `h0` is the whole first interval and the
    /// tolerance is `1e-13`, so the controller has to cut back before it accepts anything.
    #[test]
    fn the_step_counts_are_summed_over_every_arc_and_the_rejections_are_counted_too() {
        let sys = Linear::new();
        let sol = integrate_delayed(
            &Dop853,
            &sys,
            InitialValue::at_rest(0.0, [1.0]),
            8.0,
            1.0,
            dop853_controls(),
        )
        .expect("the linear delayed equation integrates");

        let (accepted, rejected) = sol.step_counts();
        let arc_accepted: u32 = sol.arcs.iter().map(|a| a.step_counts().0).sum();
        let arc_rejected: u32 = sol.arcs.iter().map(|a| a.step_counts().1).sum();
        println!(
            "delayed steps: {accepted} accepted, {rejected} rejected over {} arcs",
            sol.arcs.len()
        );
        assert_eq!((accepted, rejected), (arc_accepted, arc_rejected));
        assert!(accepted > 1, "only {accepted} accepted steps");
        assert!(
            rejected > 1,
            "only {rejected} rejected steps — the fixture stopped forcing them"
        );
    }

    /// The interior filter in `breakpoint_mesh` looks redundant — [`breakpoints`] already returns
    /// times strictly inside the span — and it is not, because it is applied after the times have
    /// been shifted back onto `t0`. At `t0 = 1e16` one ulp is 2 seconds, so a breakpoint a
    /// millisecond after `t0` *is* `t0` once it is added, and a breakpoint 99.5 seconds after it is
    /// `t1`. Both have to go: a mesh that repeats an endpoint has a zero-length arc in it, which is
    /// the `Mesh` error the second half of this test names.
    #[test]
    fn a_breakpoint_that_floating_point_absorbed_into_an_endpoint_is_dropped() {
        let t0 = 1e16;
        let t1 = 1e16 + 100.0;
        assert_eq!(t0 + 1.0e-3, t0, "the fixture needs the low end to absorb");
        assert_eq!(t0 + 99.5, t1, "the fixture needs the high end to absorb");

        let mesh =
            breakpoint_mesh(&[1.0e-3, 99.5], t0, t1).expect("absorbed breakpoints are dropped");
        assert_eq!(mesh, vec![t1]);

        // The ordinary case, where nothing absorbs and the interior breakpoints are all kept.
        let plain = breakpoint_mesh(&[0.25], 0.0, 1.0).expect("well-separated breakpoints");
        assert_eq!(plain, vec![0.25, 0.5, 0.75, 1.0]);
    }

    /// `breakpoints` is documented to return times *strictly* inside `(t0, t1)`, and the boundary is
    /// where that is decided: `t0 + 2τ` with `t1 = t0 + 2τ` is the case that separates a strict
    /// comparison from a permissive one, and it is exactly the case `breakpoint_mesh` would then
    /// have to filter out again.
    #[test]
    fn a_breakpoint_landing_on_the_final_time_is_not_inside_the_span() {
        assert_eq!(breakpoints(&[1.0], 0.0, 2.0, 4), Some(vec![1.0]));
        assert_eq!(breakpoints(&[1.0], 0.0, 2.5, 4), Some(vec![1.0, 2.0]));
    }

    /// Both endpoints are checked, and the guard is a disjunction of four independent reasons to
    /// refuse — so each has to be reached with the other three false. A NaN endpoint is the one that
    /// needs saying out loud: `t1 <= t0` is false when either is NaN, so the finiteness test is the
    /// only thing standing between a NaN and a mesh built out of it.
    #[test]
    fn a_non_finite_endpoint_is_refused_whichever_one_it_is() {
        assert_eq!(breakpoints(&[1.0], f64::NAN, 10.0, 4), None);
        assert_eq!(breakpoints(&[1.0], 0.0, f64::NAN, 4), None);
        assert_eq!(breakpoints(&[1.0], f64::NEG_INFINITY, 10.0, 4), None);
        assert_eq!(breakpoints(&[1.0], 0.0, f64::INFINITY, 4), None);
        assert_eq!(breakpoints(&[1.0], 5.0, 5.0, 4), None);
        assert_eq!(breakpoints(&[-1.0], 0.0, 10.0, 4), None);
    }

    /// The merge tolerance is `8 ε` of the *span*, not of one, and the difference only shows on a
    /// span far from one. `0.1 + 0.2` and `0.3` differ by `5.6e-17`: over a span of 1000 that is
    /// well inside `8 ε · span = 1.8e-12` and the two are the same breakpoint, while a tolerance
    /// divided by the span instead of multiplied by it would keep both and put a mesh point 56
    /// attoseconds after another one.
    #[test]
    fn the_merge_tolerance_is_scaled_by_the_span_it_is_measured_on() {
        assert_ne!(
            0.1 + 0.2,
            0.3,
            "the fixture needs the two paths to disagree"
        );
        let inner = breakpoints(&[0.1, 0.2, 0.3], 0.0, 1000.0, 2).expect("finite positive delays");
        let merged: Vec<f64> = inner
            .iter()
            .copied()
            .filter(|t| (t - 0.3).abs() < 1e-9)
            .collect();
        println!("breakpoints near 0.3: {merged:?}");
        assert_eq!(merged.len(), 1, "0.1 + 0.2 and 0.3 were not merged");
    }

    /// The cap counts the set it is about to hold, so it is `all + next` and the comparison is
    /// strict. One delay per breakpoint makes the first level land exactly on the limit, which is
    /// the only place the three ways of writing that comparison come apart: at the limit it is
    /// admitted, one past it refused, and a product in place of the sum is zero on the first level
    /// and admits anything.
    #[test]
    #[allow(clippy::cast_precision_loss)] // Counts far inside `f64`'s exact integer range.
    fn the_breakpoint_cap_admits_its_own_limit_and_refuses_one_more() {
        let at_limit: Vec<f64> = (1..=MAX_INTERVALS).map(|i| i as f64).collect();
        let span = MAX_INTERVALS as f64 + 10.0;
        assert_eq!(
            breakpoints(&at_limit, 0.0, span, 1).map(|b| b.len()),
            Some(MAX_INTERVALS)
        );

        let one_more: Vec<f64> = (1..=MAX_INTERVALS + 1).map(|i| i as f64).collect();
        assert_eq!(breakpoints(&one_more, 0.0, span, 1), None);
    }

    /// The interval cap is the same shape of comparison one level down, and the same boundary case
    /// decides it: a unit gap cut to a `1e-6` delay asks for exactly [`MAX_INTERVALS`] pieces, which
    /// is the largest mesh that is still allowed rather than the smallest that is not.
    #[test]
    #[allow(clippy::cast_precision_loss)] // Counts far inside `f64`'s exact integer range.
    fn the_interval_cap_admits_exactly_its_own_limit() {
        let tau = 1.0 / MAX_INTERVALS as f64;
        assert_eq!(mesh_pieces(0.0, &[1.0], tau), Ok(std::vec![MAX_INTERVALS]));
        assert!(matches!(
            mesh_pieces(0.0, &[1.0 + tau], tau),
            Err(DelayError::TooManyIntervals { .. })
        ));
    }

    /// The last piece ends at `end` exactly, not at `start + count · step`. Most counts hide the
    /// difference — `3 · (1/3)` rounds back to exactly `1` — so the fixture is 49 pieces, the
    /// smallest count of `[0, 1]` for which the accumulation lands an ulp short at
    /// `0.9999999999999999`. An arc that stopped there would leave the next interval's history
    /// lookup short of the join by exactly the amount the dense output is not valid over.
    #[test]
    #[allow(clippy::cast_precision_loss)] // Counts far inside `f64`'s exact integer range.
    fn the_last_piece_ends_exactly_where_the_mesh_says() {
        let count = 49;
        let step = 1.0 / count as f64;
        assert_ne!(
            count as f64 * step,
            1.0,
            "the fixture needs the accumulation to drift"
        );
        assert_eq!(piece(0.0, 1.0, count - 1, count).1, 1.0);
        assert_eq!(piece(0.0, 1.0, 0, count).0, 0.0);
        assert_eq!(piece(0.0, 1.0, 1, count).0, step);
        assert_eq!(piece(0.0, 1.0, 0, 1), (0.0, 1.0));
    }
}
