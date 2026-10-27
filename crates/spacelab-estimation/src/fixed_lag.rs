//! Fixed-lag smoothing: most of what the backward pass buys, at a bounded latency.
//!
//! [`RtsSmoother`] is fixed-*interval* — it needs the whole arc before it can
//! say anything about any epoch in it. That is the right tool for a fitted arc and the wrong one for
//! anything running. An operational orbit determination, a navigation filter, a tracker feeding a
//! catalogue: each needs an estimate now, or within a bounded delay, and each is willing to trade a
//! few epochs of latency for a covariance that conditions on data the filter had not yet seen.
//!
//! A fixed-lag smoother reports epoch `k − L` conditioned on everything through `k`: latency `L`, and
//! a covariance somewhere between the filter's and the fixed-interval smoother's. How far between is
//! measured below, and it is not the number the textbook framing suggests.
//!
//! # Why this re-runs the window instead of recursing
//!
//! The obvious implementation is the one below: keep the last `L` intervals, and each epoch run the
//! RTS backward pass over them from the current filtered estimate. That is `L` backward steps per
//! epoch, `O(L·N³)`.
//!
//! The classical alternative is a genuine recursion — augment the state with `L` delayed copies, or
//! use the Biswas–Mahalanabis update, and pay `O(N³)` per epoch with no `L` in it. It was rejected
//! on numerical grounds, not on cost: that family of recursions propagates a difference of
//! covariances across epochs and is documented to diverge (Kelly & Anderson 1971, *On the stability
//! of fixed-lag smoothing algorithms*), which is the same failure the array form of
//! [`rts`](crate::rts) exists to avoid. Re-running the window inherits that module's guarantee
//! instead: `P̂` is a Gram product at every step and nothing is ever subtracted.
//!
//! `L` is a lag, not an arc. At `L = 10` and `N = 6` the window costs ten `6×12` triangularisations
//! per epoch, which is cheaper than one geopotential evaluation. Paying `O(L)` to keep the estimate
//! provably positive-definite is not a close call.
//!
//! # What a lag buys, measured
//!
//! `tests/fixed_lag_smoothing.rs` sweeps the lag on a six-state constant-velocity arc, position
//! observed on three axes, at two levels of unmodelled acceleration `q`. The row that matters is the
//! fraction of the *fixed-interval* gain each lag captures, in reported position sigma:
//!
//! ```text
//!                 lag      0      1      2      3      5     10     20     40    RTS
//!   q = 1e-4  σ (m)    4.486  4.330  4.183  4.043  3.787  3.253  2.528  1.858  1.705
//!             of RTS     0 %  5.6 % 10.9 % 15.9 % 25.2 % 44.3 % 70.4 % 94.5 %  100 %
//!   q = 3e-1  σ (m)   14.976 10.528 10.380 10.295 10.158 10.154 10.154 10.154 10.154
//!             of RTS     0 % 92.2 % 95.3 % 97.1 % 99.9 %  100 %  100 %  100 %  100 %
//! ```
//!
//! **The useful lag is set by the ratio of process noise to measurement noise, and the two regimes
//! are nothing alike.** On the noisy arc — a manoeuvring target, or a force model that does not fit —
//! a lag of *one* captures 92 % and a lag of five is at the limit. On the well-modelled arc a lag of
//! five captures 25 %, and it takes a lag of forty to reach 95 %.
//!
//! The reason is that the smoother gain `C = P⁺ΦᵀP⁻⁻¹` decays at the rate the filter forgets. A
//! filter with almost no process noise never forgets, so every future observation still says
//! something about the past and the useful lag is the whole arc.
//!
//! The operational consequence is worth stating plainly, because the textbook framing (which assumes
//! a manoeuvring target) points the other way: **for precise orbit determination on a well-modelled
//! arc, a short-lag smoother is close to pointless — use the fixed-interval pass.** Fixed-lag earns
//! its latency where the dynamics are poorly known, which is exactly where an operational filter is
//! usually running anyway.
//!
//! NEES sits at 6.1–6.3 against six degrees of freedom at every lag on both arcs, so the covariance
//! shrinks honestly rather than by losing a subtraction.
//!
//! At `LAG = 0` this is filtering, exactly and by construction, which is the boundary case a lag
//! sweep needs at one end.
//!
//! Gate: `derivations/rts_smoother.mac` evaluates "epoch `k − L` conditioned on `z₀…z_k`" as a
//! joint-Gaussian conditional in exact rationals, and `derivations/linear_filters.gp` recomputes it.
//! The test `tests/fixed_lag_smoothing.rs::the_fixed_lag_estimate_is_the_conditional_on_the_data_so_far`
//! (G2) holds a lag-2 smoother to it at two epochs and (G3) a lag longer than the arc to the
//! fixed-interval answer.

// Matrix code indexes several arrays by one i/j/k, and the Kalman symbols are one letter by
// universal convention.
#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]

use crate::rts::{RtsError, RtsSmoother, SmoothingStep};

/// A ring of the last `LAG` recorded intervals, and the backward pass over them.
///
/// `LAG` is the reporting delay in epochs: [`smooth`](Self::smooth) returns the estimate at the epoch
/// `LAG` intervals behind the one the caller is filtering now. Before the ring fills, the lag is
/// whatever history exists — see [`depth`](Self::depth) — so the smoother is useful from the second
/// epoch rather than from the `LAG`-th.
#[derive(Clone, Debug)]
pub struct FixedLagSmoother<const N: usize, const LAG: usize> {
    window: [SmoothingStep<N>; LAG],
    /// Where the next interval will be written.
    next: usize,
    /// How many of the slots hold a recorded interval, saturating at `LAG`. Every read is bounded by
    /// this, so the unwritten slots are never reached and need no sentinel.
    depth: usize,
}

impl<const N: usize, const LAG: usize> Default for FixedLagSmoother<N, LAG> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const LAG: usize> FixedLagSmoother<N, LAG> {
    /// A window whose slots are zero-filled and, because [`depth`](Self::depth) starts at zero,
    /// unreachable. The alternative — `Option<SmoothingStep<N>>` — would put a `None` on a path that
    /// the ring's own bookkeeping already makes impossible, and library code cannot `unwrap` it.
    const EMPTY: SmoothingStep<N> = SmoothingStep {
        filtered: [0.0; N],
        filtered_sqrt: [[0.0; N]; N],
        stm: [[0.0; N]; N],
        process_noise_sqrt: [[0.0; N]; N],
        predicted: [0.0; N],
    };

    /// An empty window.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            window: [Self::EMPTY; LAG],
            next: 0,
            depth: 0,
        }
    }

    /// Record the interval leaving the epoch the filter has just finished, overwriting the oldest.
    ///
    /// Call once per epoch, after that epoch's measurement update and once the caller has propagated
    /// the state to the next one — [`SmoothingStep::predicted`] is the propagated state, which is the
    /// one quantity the filter never sees.
    pub fn push(&mut self, step: SmoothingStep<N>) {
        if LAG == 0 {
            return;
        }
        self.window[self.next] = step;
        self.next = (self.next + 1) % LAG;
        if self.depth < LAG {
            self.depth += 1;
        }
    }

    /// How many intervals the next [`smooth`](Self::smooth) will step back over: `LAG` once the ring
    /// has filled, and the number of recorded intervals before that.
    ///
    /// This is the *actual* lag of the estimate being reported, and it is not `LAG` at the start of
    /// an arc. A caller time-tagging the output needs this number, not the constant.
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// The smoothed estimate [`depth`](Self::depth) intervals back, conditioned on everything through
    /// the current epoch.
    ///
    /// `filtered` and `filtered_sqrt` are the filter's own output at the current epoch — the seed of
    /// the backward pass, where smoothed and filtered coincide because there is no future left.
    ///
    /// # Errors
    ///
    /// [`RtsError::SingularPrediction`] if any interval in the window has a singular predicted
    /// covariance, reported with the direction, exactly as [`RtsSmoother::step`] would.
    pub fn smooth(
        &self,
        filtered: &[f64; N],
        filtered_sqrt: &[[f64; N]; N],
    ) -> Result<([f64; N], [[f64; N]; N]), RtsError> {
        let mut smoother = RtsSmoother::new(*filtered, *filtered_sqrt);
        for back in 1..=self.depth {
            // Walk backwards from the most recent write. `depth ≤ LAG` bounds this to slots that
            // `push` has written, so the ring never yields an `EMPTY`.
            let index = (self.next + LAG - back) % LAG;
            smoother.step(&self.window[index])?;
        }
        Ok((*smoother.state(), smoother.covariance_sqrt()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp, clippy::cast_precision_loss)]

    use super::*;

    fn interval(index: usize) -> SmoothingStep<2> {
        let t = index as f64;
        SmoothingStep {
            filtered: [t, 0.5 * t],
            filtered_sqrt: [[1.0 + 0.1 * t, 0.0], [0.2, 0.7]],
            stm: [[1.0, 1.0], [0.0, 1.0]],
            process_noise_sqrt: [[0.1, 0.0], [0.0, 0.05]],
            predicted: [t + 0.5 * t, 0.5 * t],
        }
    }

    #[test]
    fn a_zero_lag_smoother_is_the_filter() {
        let mut smoother = FixedLagSmoother::<2, 0>::new();
        for i in 0..5 {
            smoother.push(interval(i));
        }
        assert_eq!(smoother.depth(), 0);
        let seed = ([3.0, -1.0], [[2.0, 0.0], [0.5, 1.0]]);
        let (x, s) = smoother.smooth(&seed.0, &seed.1).unwrap();
        assert_eq!((x, s), seed);
    }

    #[test]
    fn the_depth_is_the_history_not_the_constant() {
        let mut smoother = FixedLagSmoother::<2, 3>::new();
        assert_eq!(smoother.depth(), 0);
        for (pushed, expected) in [(0, 1), (1, 2), (2, 3), (3, 3), (4, 3)] {
            smoother.push(interval(pushed));
            assert_eq!(smoother.depth(), expected);
        }
    }

    /// The ring has to hand the backward pass the *most recent* `depth` intervals, newest first, and
    /// getting that wrong is silent: the smoother still returns a plausible number.
    ///
    /// Pinned by running a full [`RtsSmoother`] over the same intervals in the same order by hand.
    #[test]
    fn the_window_replays_the_newest_intervals_newest_first() {
        let mut smoother = FixedLagSmoother::<2, 3>::new();
        for i in 0..6 {
            smoother.push(interval(i));
        }
        let seed = ([7.0, 1.5], [[1.5, 0.0], [0.3, 0.9]]);
        let (x, s) = smoother.smooth(&seed.0, &seed.1).unwrap();

        let mut expected = RtsSmoother::new(seed.0, seed.1);
        for i in [5, 4, 3] {
            expected.step(&interval(i)).unwrap();
        }
        assert_eq!(x, *expected.state());
        assert_eq!(s, expected.covariance_sqrt());
    }

    #[test]
    fn a_singular_prediction_is_reported_rather_than_smoothed_over() {
        let mut degenerate = interval(0);
        degenerate.filtered_sqrt = [[0.0; 2]; 2];
        degenerate.process_noise_sqrt = [[0.0; 2]; 2];
        let mut smoother = FixedLagSmoother::<2, 2>::new();
        smoother.push(degenerate);
        let error = smoother
            .smooth(&[0.0, 0.0], &[[1.0, 0.0], [0.0, 1.0]])
            .unwrap_err();
        assert!(matches!(error, RtsError::SingularPrediction { .. }));
    }
}
