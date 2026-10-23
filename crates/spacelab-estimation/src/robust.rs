//! Outlier-robust measurement updates: χ² gating, Huber M-estimation, and a Student-t variational
//! update.
//!
//! Tracking data contains gross errors — a mis-associated observation in a crowded field, a cosmic
//! ray on the CCD, radar multipath, a mislabelled epoch. A Kalman filter has no defence: the update
//! is linear in the innovation, so one bad observation moves the estimate in proportion to how bad it
//! is, and *shrinks the covariance by the same amount it would have for a good one*. The filter comes
//! out confidently wrong, which is the failure mode that is hardest to read off a residual plot.
//!
//! # One primitive, three policies
//!
//! Every robust scheme here is a rule for one number: the **effective measurement-noise variance**
//! `R_eff` this observation is admitted with. That is deliberate. The square-root update in
//! [`SquareRootKalman::measurement_update`] is a single orthogonal transformation parameterised by
//! `R`, so re-weighting an observation and admitting it with an inflated `R` are the same operation —
//! and doing it that way means there is exactly one update path in the crate. A second, separately
//! written robust update is a second thing to keep correct, and the two would drift.
//!
//! Writing `p = h P⁻ hᵀ` for the prior variance in measurement space, `R` for the nominal noise
//! variance and `ν` for the innovation, the nominal normalised innovation squared is `δ² = ν²/(p+R)`:
//!
//! * [`ChiSquareGate`](RobustPolicy::ChiSquareGate) — reject outright when `δ²` exceeds a quantile of
//!   χ²(1). Bounded influence by construction, and discontinuous: an observation just inside the gate
//!   enters at full weight and one just outside is thrown away.
//! * [`Huber`](RobustPolicy::Huber) — quadratic loss inside the tuning constant `c`, linear outside,
//!   applied as an iteratively re-weighted update. `R_eff = R/w` with `w = min(1, c/|δ|)`, and `δ` is
//!   itself measured against `R_eff` — a fixed point, solved exactly below rather than taken after one
//!   step.
//! * [`StudentT`](RobustPolicy::StudentT) — model the heavy tail instead of clipping it. One
//!   variational-Bayes step for Student-t measurement noise, which for a scalar observation has a
//!   closed form (no iteration at all). This is the one to prefer when the outlier rate is unknown,
//!   because it needs no threshold.
//!
//! # Which one to use, measured
//!
//! `tests/robust_estimation.rs` runs all four against a plain Kalman filter on the same contaminated
//! tracking arc — 100 runs × 300 steps × 3 axes per cell, every estimator seeing byte-identical data.
//! RMS position error in metres, contamination by `±20σ` outliers:
//!
//! ```text
//!                    0 %       1 %       5 %      20 %
//!     plain          4.41      9.96     20.14     37.84
//!     gate 3σ        4.51    931.97   1644.27   1894.05
//!     inflate 3σ     4.42      6.09     14.45     74.47
//!     Huber 1.345    4.48      5.92     11.58     22.29
//!     Student-t ν=5  4.52      5.86     11.53     22.73
//! ```
//!
//! **Use [`Huber`](RobustPolicy::Huber) or [`StudentT`](RobustPolicy::StudentT).** They are within 2 %
//! of each other at every contamination level, they cost 1.4 % and 2.4 % on clean data, and they are
//! the only two that still work at 20 %.
//!
//! That is not the ranking the influence functions predict, and the reason is worth understanding
//! before choosing a policy for a different problem.
//!
//! # The influence function is the wrong thing to optimise
//!
//! How far can one observation move the state as `|ν| → ∞`?
//!
//! ```text
//!     gate       →  0             rejected outright past the threshold
//!     inflate    →  0             redescending: R_eff ∝ ν², so the increment falls as 1/|ν|
//!     Student-t  →  p√((n+1)/R)   bounded, and it saturates there
//!     Huber      →  ∝ |ν|^(1/3)   *not* bounded — sublinear, but unbounded
//! ```
//!
//! By that measure Huber is the worst of the four and the gate is the best. The measurement says the
//! reverse, and the reason is that resisting one bad observation is not the problem. **Recovering
//! from a bad state is.** A filter whose estimate is already wrong sees good observations as
//! outliers, and an estimator that refuses to move for a large innovation then refuses to move at
//! all. Huber's unbounded influence — the property that looks like a defect — is exactly what lets it
//! climb back.
//!
//! **That inversion is conditional, and `tests/robust_vector_estimation.rs` is what pins the
//! condition.** Run on a star tracker whose gyro reinflates `P` by the whole steady-state variance
//! every second, the ranking is obeyed to the letter instead: gate 53.3″, Student-t 507″, inflation
//! 612″, Huber 660″, plain 5170″ at 20 % contamination. Same policies, same code, opposite order. The
//! influence function ranks correctly exactly when the process noise can undo a mistake, and backwards
//! when it cannot — so the quantity to look at before choosing a policy is `Q` against `R`, not the
//! estimator. That file also measures the sharper version of the lockout: with **no outliers at all**,
//! a gated MEKF started 2.9° from truth rejects 62.7 % of its data and settles 440× worse than the
//! plain filter, because a linearisation transient is indistinguishable from contamination.
//!
//! The gate is the extreme case, and its failure is total. Five whys, from the measurement:
//!
//! 1. RMS 932 m at 1 % contamination. Why? The state drifts unchecked for the whole arc.
//! 2. Why unchecked? 299 of 300 steps rejected every observation on that axis.
//! 3. Why rejected? Every innovation exceeded `3√f`.
//! 4. Why? The state was already biased by an outlier taken earlier, and the bias exceeds the gate.
//! 5. Why was that outlier accepted? It arrived while `P₀` was still loose — `√f ≈ 100 m`, so the 3σ
//!    gate was 300 m wide and a 200 m outlier walked through it.
//!
//! **The gate scales with the filter's own confidence, so it is widest exactly when the filter is
//! least able to judge, and once a bad update is taken the same mechanism that should catch the error
//! instead protects it.** [`ChiSquareInflation`](RobustPolicy::ChiSquareInflation) removes step 2 —
//! its weight is never zero — which is why it holds to 5 %; it still degrades at 20 % because
//! redescending is lockout in slow motion.
//!
//! # Gating is reported, never silent
//!
//! [`RobustTally`] exists because a filter that quietly discards a third of its data and shows a
//! beautiful residual plot is worse than one that diverges honestly. Every update returns a
//! [`RobustOutcome`] carrying the weight it applied and the normalised innovation it saw.
//!
//! The diagnostic that catches the lockout above is
//! [`longest_rejection_run`](RobustTally::longest_rejection_run), and **nothing else does**. The
//! rejection *rate* at 5 % contamination was 7.4 % against an expected 5.3 % — unremarkable. The NEES
//! was 17.2 against the plain filter's 23.6, so the locked-out filter looked *more* consistent than
//! the working one, because a state drifting linearly under a covariance growing quadratically stays
//! superficially calibrated. Only the run length showed it, at 299.
//!
// The estimation symbols (P, S, K, H, R, p, f, w, ν) are one letter by convention.
#![allow(clippy::many_single_char_names)]

use crate::SquareRootKalman;
use crate::consistency::chi_square_mean_interval;

/// How an observation whose innovation is too large for its covariance should be admitted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RobustPolicy {
    /// No robustness — the plain Kalman update. The control case, and the right choice when the
    /// measurement noise really is Gaussian.
    Plain,
    /// Reject the observation when `δ² > threshold`. Build the threshold with [`chi_square_gate`]
    /// rather than picking a number.
    ///
    /// **This policy can lock the filter out permanently, and does so in practice.** An outlier that
    /// arrives while the covariance is still loose passes the gate and biases the state; the
    /// covariance then contracts around the wrong answer, every subsequent observation looks like an
    /// outlier, and the only thing that can widen the covariance again is process noise at `Q` per
    /// step. `tests/robust_estimation.rs` measures 932 m RMS at 1 % contamination against 4.5 m on
    /// clean data — a *worse* filter than no robustness at all. Use
    /// [`ChiSquareInflation`](RobustPolicy::ChiSquareInflation) instead, or watch
    /// [`RobustTally::longest_rejection_run`], which is what makes the lockout visible.
    ChiSquareGate {
        /// Normalised-innovation-squared above which the observation is discarded.
        threshold: f64,
    },
    /// The same threshold, but inflating the noise by `δ²/threshold` instead of rejecting — the
    /// Mahalanobis-distance robust update (Chang 2014). Continuous, and it never sets the weight to
    /// zero, so it cannot lock out the way [`ChiSquareGate`](RobustPolicy::ChiSquareGate) does.
    ///
    /// It is *redescending*: `R_eff ∝ ν²` makes the state increment fall as `1/|ν|`, so a gross
    /// outlier moves the estimate less than a moderate one does. That is the strongest possible
    /// rejection of a bad observation short of discarding it, and it is also why recovery from a
    /// large existing bias is slow — the filter has to climb back down its own descending arm.
    ChiSquareInflation {
        /// Normalised-innovation-squared above which the noise starts to be inflated.
        threshold: f64,
    },
    /// Huber M-estimation with tuning constant `c` on the standardised innovation. `c = 1.345` is the
    /// classical choice — 95% asymptotic efficiency against clean Gaussian data.
    Huber {
        /// Where the loss stops being quadratic, in standard deviations of the innovation.
        tuning: f64,
    },
    /// Student-t measurement noise with `degrees_of_freedom` heavy-tail parameter, admitted by one
    /// variational-Bayes step. Small values are heavy-tailed (`4` is aggressive, `10` mild); as it
    /// grows the policy tends to [`Plain`](RobustPolicy::Plain).
    ///
    /// The `noise_variance` passed to the update is read as the Student-t **scale**, not its variance
    /// — a `t(n)` with scale `s` has variance `s·n/(n−2)`. Pass the nominal `R` anyway: the effective
    /// variance this policy produces is `s(n + δ²)/(n+1)`, and on clean data `E[δ²] ≈ 1`, so
    /// `E[R_eff] ≈ s`. The scale is the *mean* effective noise, which is the quantity that has to be
    /// right for the covariance to be realistic.
    ///
    /// Variance-matching instead — passing `R(n−2)/n`, which looks like the careful thing to do —
    /// was measured to leave the filter 40 % overconfident: NEES 9.4 against 6.2 on six degrees of
    /// freedom, on clean data, for no gain in error.
    StudentT {
        /// The tail parameter. Must exceed 0; below 2 the distribution has no variance, which is a
        /// legitimate model but makes the scale/variance conversion above meaningless.
        degrees_of_freedom: f64,
    },
}

/// The χ² gate threshold for `dof` degrees of freedom at standard-normal quantile `z` (1.96 → 95%,
/// 2.576 → 99%, 3.0 → 99.7%).
///
/// This is [`chi_square_mean_interval`] with a single sample — the same Wilson–Hilferty cube the
/// consistency diagnostics use, so the gate and the NIS test that audits it cannot disagree about
/// what χ² means. That reuse costs some accuracy at one degree of freedom, where the cube-root
/// approximation is at its worst: measured against the tables it returns 4.93, 7.91 and 10.53 for
/// those three quantiles against the exact 5.02, 7.88 and 10.28 — −1.9 %, +0.3 %, +2.5 %. By
/// `dof = 3` it is within 0.3 %.
///
/// For the scalar case there is an exact alternative that needs no table at all: rejecting when
/// `|ν| > k√f` is `threshold = k²`, with two-sided coverage `2Φ(k) − 1` — a 3σ gate is `9.0`. Prefer
/// it when the intent is "how many sigmas", and this function when the intent is "what tail".
#[must_use]
pub fn chi_square_gate(dof: usize, z: f64) -> f64 {
    chi_square_mean_interval(dof, 1, z).1
}

/// What a robust update did with one observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustOutcome {
    /// `false` when the observation was rejected outright and the state and covariance are unchanged.
    pub accepted: bool,
    /// `R / R_eff` — 1 for a fully admitted observation, 0 for a rejected one. Below 1 the observation
    /// was admitted with inflated noise; **above** 1 is possible only under
    /// [`StudentT`](RobustPolicy::StudentT), whose scale is not its variance.
    pub weight: f64,
    /// `δ² = ν²/(p + R)` against the *nominal* noise, which is what a NIS diagnostic would see. The
    /// quantity to log, and the one to compare against [`chi_square_gate`].
    pub nominal_normalized_squared: f64,
    /// The innovation variance the update actually ran with, `p + R_eff`. Zero when rejected.
    ///
    /// A vector update has a matrix there and this field is a scalar, so the vector entry points
    /// report `tr(S_eff)/d`, the mean per-axis innovation variance. Read it as a magnitude, not as a
    /// quantity to divide an innovation by — [`nominal_normalized_squared`](Self::nominal_normalized_squared)
    /// is the one that is comparable across the scalar and vector paths.
    pub innovation_variance: f64,
}

/// A running census of what a robust filter did to its data.
///
/// Report it. The point of the whole module is that the decision to discount an observation is a
/// decision, and a filter that makes thousands of them silently has replaced a visible divergence
/// with an invisible bias.
///
/// **One tally per measurement stream.** [`longest_rejection_run`](Self::longest_rejection_run) is a
/// per-stream statistic and pooling streams destroys it: the same lockout that shows as a run of 299
/// in a per-axis tally shows as 14 when three interleaved axes share one, because the other two axes
/// keep breaking the run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RobustTally {
    /// Observations offered to the filter.
    pub seen: u32,
    /// Observations rejected outright.
    pub rejected: u32,
    /// Observations admitted with `weight < 1` — down-weighted but used.
    pub downweighted: u32,
    /// The longest unbroken run of rejections.
    ///
    /// This is the diagnostic that matters, and it is not the same information as
    /// [`rejected`](Self::rejected). A gating filter that has locked out — biased state, contracted
    /// covariance, every subsequent observation outside the gate — rejects *everything from then on*,
    /// and a rejection rate of 8 % against an expected 5 % is easy to shrug at while a run of 400
    /// consecutive rejections is not. Compare it against how many rejections in a row the
    /// contamination rate can plausibly produce: at 5 % contamination a run of 10 has probability
    /// `1e-13`.
    pub longest_rejection_run: u32,
    current_rejection_run: u32,
    weight_sum: f64,
}

impl RobustTally {
    /// Fold in one update's outcome.
    pub fn record(&mut self, outcome: &RobustOutcome) {
        self.seen += 1;
        if outcome.accepted {
            self.weight_sum += outcome.weight;
            if outcome.weight < 1.0 {
                self.downweighted += 1;
            }
            self.current_rejection_run = 0;
        } else {
            self.rejected += 1;
            self.current_rejection_run += 1;
            self.longest_rejection_run = self.longest_rejection_run.max(self.current_rejection_run);
        }
    }

    /// Mean weight over every observation seen, counting a rejection as zero. `1` means the data was
    /// treated as clean throughout; a number far below the contamination fraction the caller expects
    /// means the filter is discarding good data.
    #[must_use]
    pub fn mean_weight(&self) -> f64 {
        if self.seen == 0 {
            return 0.0;
        }
        self.weight_sum / f64::from(self.seen)
    }

    /// Fraction of observations rejected outright.
    #[must_use]
    pub fn rejected_fraction(&self) -> f64 {
        if self.seen == 0 {
            return 0.0;
        }
        f64::from(self.rejected) / f64::from(self.seen)
    }
}

/// The Huber weight `w = min(1, c/|δ|)` at its own fixed point, where `δ² = ν²/(p + R/w)`.
///
/// Not the one-step form. Substituting the weight back into the innovation variance it scales gives
/// `w³ν² − c²p·w − c²R = 0`, whose coefficients have exactly one sign change, so by Descartes there
/// is exactly one positive root — and `h(0) = −c²R < 0` with `h(1) = (p+R)(δ₀² − c²) > 0` whenever the
/// observation is outside the tuning constant, so that root is in `(0, 1)` and bisection finds it
/// without a starting guess or a convergence assumption.
fn huber_weight(prior: f64, noise: f64, innovation: f64, tuning: f64) -> f64 {
    let nominal = prior + noise;
    let squared = innovation * innovation;
    if squared <= tuning * tuning * nominal {
        return 1.0;
    }
    let c2 = tuning * tuning;
    let residual = |w: f64| w * w * w * squared - c2 * prior * w - c2 * noise;
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    // 60 halvings takes a unit interval below f64 resolution; the loop is a fixed cost, not a search.
    for _ in 0..60 {
        let mid = f64::midpoint(lo, hi);
        if residual(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

/// The effective noise variance of one variational-Bayes step for Student-t measurement noise.
///
/// The E-step gives the expected precision scale `λ = (n+1)/(n + δ²)` with `δ² = ν²/f` and
/// `f = p + R/λ`. Those three eliminate to a quadratic in `f`,
///
/// ```text
///     (n+1) f² − [(n+1)p + nR] f − Rν² = 0,
/// ```
///
/// whose positive root is closed-form — no iteration, and no cancellation, since the constant term is
/// negative and the two terms under the root therefore add. `R_eff = f − p`.
///
/// Two consequences worth stating. At `ν = 0` it returns `R·n/(n+1)`, *less* than nominal, because the
/// Student-t scale is not its variance. And as `|ν| → ∞`, `f → |ν|√(R/(n+1))`, so the gain falls as
/// `1/|ν|` exactly fast enough for the state increment `Kν` to converge — the bounded influence
/// function that neither of the other two policies has.
fn student_t_variance(prior: f64, noise: f64, innovation: f64, dof: f64) -> f64 {
    let a = dof + 1.0;
    let b = a * prior + noise * dof;
    let discriminant = b * b + 4.0 * a * noise * innovation * innovation;
    let f = (b + libm::sqrt(discriminant)) / (2.0 * a);
    f - prior
}

impl RobustPolicy {
    /// The effective noise variance this policy admits the observation with, or `None` to reject it.
    ///
    /// `prior_variance` is `h P⁻ hᵀ`, the prior uncertainty projected onto the measurement. Returns
    /// `None` for a non-physical input (negative variance, non-finite innovation) as well as for a
    /// gated observation: an update that cannot be posed is not one to guess at.
    #[must_use]
    pub fn effective_variance(
        &self,
        prior_variance: f64,
        noise_variance: f64,
        innovation: f64,
    ) -> Option<f64> {
        if !prior_variance.is_finite()
            || !noise_variance.is_finite()
            || !innovation.is_finite()
            || prior_variance < 0.0
            || noise_variance <= 0.0
        {
            return None;
        }
        let nominal = prior_variance + noise_variance;
        match *self {
            Self::Plain => Some(noise_variance),
            Self::ChiSquareGate { threshold } => {
                (innovation * innovation <= threshold * nominal).then_some(noise_variance)
            }
            Self::ChiSquareInflation { threshold } => (threshold > 0.0).then(|| {
                let normalized = innovation * innovation / nominal;
                noise_variance
                    * if normalized > threshold {
                        normalized / threshold
                    } else {
                        1.0
                    }
            }),
            Self::Huber { tuning } => (tuning > 0.0).then(|| {
                noise_variance / huber_weight(prior_variance, noise_variance, innovation, tuning)
            }),
            Self::StudentT { degrees_of_freedom } => (degrees_of_freedom > 0.0).then(|| {
                student_t_variance(
                    prior_variance,
                    noise_variance,
                    innovation,
                    degrees_of_freedom,
                )
            }),
        }
    }

    /// The factor an **isotropic vector** observation's noise `R = σ²I` should be multiplied by, or
    /// `None` to reject the observation. `normalized_squared` is `δ² = νᵀS⁻¹ν` formed against the
    /// *nominal* `S = HPHᵀ + R`, and `dof` is the dimension of `ν`.
    ///
    /// **This is the one-step form, not the fixed point [`effective_variance`](Self::effective_variance)
    /// solves — the two are not equivalent and must not be assumed to be.** For a scalar the weight can
    /// be substituted back into the variance it scales and the resulting cubic solved exactly, because
    /// `f = p + R/w` is one number. For a vector `S = HPHᵀ + R/w`, and unless `HPHᵀ` and `R` share
    /// eigenvectors no scalar `w` reproduces the Mahalanobis norm at all — the fixed point is a matrix
    /// equation in a matrix unknown. The standard answer (Chang 2014) is to measure `δ²` once against
    /// the nominal `S` and scale from there without re-solving, which is what this does.
    ///
    /// `R` isotropic is a precondition, not a convenience: every vector update in this crate takes one
    /// `sigma` for all axes, so a scalar factor is exact there. Applied to an anisotropic `R` it
    /// discounts the well-measured axes by as much as the badly-measured ones.
    ///
    /// Two of the four policies need `dof` for more than the gate threshold:
    ///
    /// * **Huber** weights on `δ̄ = √(δ²/d)`, the per-degree-of-freedom radius, whose expectation is 1
    ///   at every `d`. Weighting on `√δ²` instead would leave `c = 1.345` meaning "95 % efficiency" at
    ///   `d = 1` and "down-weight two thirds of clean data" at `d = 3`, since `E[δ²] = d`.
    /// * **Student-t** uses the multivariate VB E-step `λ = (n + d)/(n + δ²)`, which is exactly 1 at
    ///   `δ² = d` and reduces to the scalar `(n + 1)/(n + δ²)` at `d = 1`.
    ///
    /// The χ² policies are unchanged in form — build their threshold with [`chi_square_gate`] at the
    /// same `dof`, where Wilson–Hilferty is *more* accurate than in the scalar case, not less. At
    /// `dof = 1` those three reproduce [`effective_variance`](Self::effective_variance) exactly.
    ///
    /// The approximation has a **direction**, and it is worth knowing before porting a tuning constant
    /// across: the fixed point measures `δ` against the inflated variance, this measures it once
    /// against the nominal one, and the nominal one is smaller. So the one-step form always sees a
    /// larger standardised residual and discounts **harder** than the fixed point it approximates —
    /// never more softly. A `c` tuned on a scalar filter gives a more aggressive vector filter.
    #[must_use]
    pub fn variance_scale(&self, normalized_squared: f64, dof: usize) -> Option<f64> {
        if !normalized_squared.is_finite() || normalized_squared < 0.0 || dof == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        // dof is a measurement dimension, never above a handful
        let d = dof as f64;
        match *self {
            Self::Plain => Some(1.0),
            Self::ChiSquareGate { threshold } => (normalized_squared <= threshold).then_some(1.0),
            Self::ChiSquareInflation { threshold } => {
                (threshold > 0.0).then(|| (normalized_squared / threshold).max(1.0))
            }
            Self::Huber { tuning } => {
                (tuning > 0.0).then(|| (libm::sqrt(normalized_squared / d) / tuning).max(1.0))
            }
            Self::StudentT { degrees_of_freedom } => (degrees_of_freedom > 0.0)
                .then(|| (degrees_of_freedom + normalized_squared) / (degrees_of_freedom + d)),
        }
    }
}

/// What a rejected observation leaves behind: state and covariance unchanged, and the `δ²` that
/// decided it. Shared by every vector update so the four filters cannot disagree about what a
/// rejection reports.
pub(crate) const fn rejected(normalized_squared: f64) -> RobustOutcome {
    RobustOutcome {
        accepted: false,
        weight: 0.0,
        nominal_normalized_squared: normalized_squared,
        innovation_variance: 0.0,
    }
}

/// A scalar measurement update under a [`RobustPolicy`].
///
/// Identical to [`SquareRootKalman::measurement_update`] except that the noise variance it runs with
/// is the policy's, and a gated observation is not applied at all. `h` is the partial `∂z/∂x`,
/// `innovation` is `z − ẑ` (the caller forms `ẑ`, so a nonlinear measurement is handled by passing its
/// predicted value), and `noise_variance` is the nominal `R`.
///
/// Feed the result to a [`RobustTally`].
pub fn robust_update<const N: usize>(
    filter: &mut SquareRootKalman<N>,
    policy: RobustPolicy,
    h: &[f64; N],
    innovation: f64,
    noise_variance: f64,
) -> RobustOutcome {
    // p = hᵀ P h = ‖Sᵀh‖², from the factor rather than from P — the same square-root discipline as
    // every other consumer of this filter.
    let s = filter.covariance_sqrt();
    let mut projected = [0.0; N];
    for (row, &weight) in s.iter().zip(h.iter()) {
        for (component, &value) in projected.iter_mut().zip(row.iter()) {
            *component += weight * value;
        }
    }
    let prior: f64 = projected.iter().map(|v| v * v).sum();
    let nominal = prior + noise_variance;
    let normalized = if nominal > 0.0 {
        innovation * innovation / nominal
    } else {
        f64::INFINITY
    };

    match policy.effective_variance(prior, noise_variance, innovation) {
        Some(effective) => {
            let used = filter.measurement_update(h, innovation, effective);
            RobustOutcome {
                accepted: true,
                weight: noise_variance / effective,
                nominal_normalized_squared: normalized,
                innovation_variance: used,
            }
        }
        None => RobustOutcome {
            accepted: false,
            weight: 0.0,
            nominal_normalized_squared: normalized,
            innovation_variance: 0.0,
        },
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

    use super::*;

    #[test]
    fn the_gate_threshold_is_the_tabulated_chi_square_quantile() {
        // The upper edge of the interval is the Φ(z)-quantile, so for one degree of freedom the exact
        // value is `(Φ⁻¹((1 + Φ(z))/2))²`: 5.0239, 7.8794, 10.2766 at these three `z`. Wilson–Hilferty
        // is a cube-root approximation and one degree of freedom is where it is weakest, so the
        // tolerance below is its accuracy, not the code's — and it is documented on the function.
        for (z, exact) in [(1.96, 5.023_886), (2.576, 7.879_400), (3.0, 10.276_618)] {
            let gate = chi_square_gate(1, z);
            assert!(
                (gate / exact - 1.0).abs() < 0.03,
                "χ²₁ at z = {z}: {gate} against {exact}"
            );
        }
        // Three degrees of freedom at the 0.975 quantile: 9.3484, and the approximation is 40× better.
        assert!((chi_square_gate(3, 1.96) / 9.348_404 - 1.0).abs() < 0.005);
        // Monotone in both arguments, which a gate has to be to mean anything.
        assert!(chi_square_gate(1, 1.96) < chi_square_gate(1, 3.0));
        assert!(chi_square_gate(1, 1.96) < chi_square_gate(3, 1.96));
    }

    #[test]
    fn a_clean_observation_is_admitted_at_full_weight() {
        let (prior, noise) = (4.0, 1.0);
        // Half a standard deviation of the innovation — nothing a filter should react to.
        let innovation = 0.5 * libm::sqrt(prior + noise);
        for policy in [
            RobustPolicy::Plain,
            RobustPolicy::ChiSquareGate { threshold: 9.0 },
            RobustPolicy::ChiSquareInflation { threshold: 9.0 },
            RobustPolicy::Huber { tuning: 1.345 },
        ] {
            let effective = policy
                .effective_variance(prior, noise, innovation)
                .expect("a clean observation is not gated");
            assert!((effective - noise).abs() < 1e-12, "{policy:?} moved R");
        }
        // The Student-t is the exception, and by an amount its own algebra fixes: at `ν = 0` exactly
        // `R·n/(n+1)`, because the scale of a t is not its variance.
        let t = RobustPolicy::StudentT {
            degrees_of_freedom: 5.0,
        };
        let at_zero = t.effective_variance(prior, noise, 0.0).unwrap();
        assert!((at_zero - noise * 5.0 / 6.0).abs() < 1e-12, "{at_zero}");
    }

    #[test]
    fn the_huber_weight_satisfies_its_own_defining_equation() {
        let (prior, noise, tuning) = (4.0, 1.0, 1.345);
        for innovation in [4.0, 5.0, 10.0, 100.0, 1e4] {
            let effective = RobustPolicy::Huber { tuning }
                .effective_variance(prior, noise, innovation)
                .unwrap();
            let w = noise / effective;
            assert!(w < 1.0, "ν = {innovation} should be down-weighted");
            // The fixed point: w = c/|δ| with δ measured against the variance w itself produces.
            let delta = innovation / libm::sqrt(prior + effective);
            assert!(
                (w - tuning / delta.abs()).abs() < 1e-12,
                "ν = {innovation}: w = {w}, c/|δ| = {}",
                tuning / delta.abs()
            );
        }
    }

    /// Where the weight stops being 1, measured from both sides of the boundary rather than from far
    /// away on either side.
    ///
    /// [`huber_weight`] returns early at `ν² ≤ c²(p + R)` and bisects otherwise, so everything the
    /// nominal variance and the two factors of `c` do is place that one edge. Sampling the two regimes
    /// decades apart never says where it is: a threshold off by a factor of two, or by `c` added
    /// instead of multiplied, still returns 1 for a quiet observation and still down-weights a gross
    /// one. Bracketing it to a part in a thousand is what makes the edge itself the thing under test.
    ///
    /// Only an edge that moves *outward* is visible, and that is why the last two cases have
    /// `p + R < 1` and `c < 1`. Moving it inward changes nothing at all: below the true threshold the
    /// cubic's root is at or above 1, so the bisection the shortcut was avoiding returns exactly 1.0
    /// anyway. `c²(p+R)` and `c² + (p+R)` and `c²/(p+R)` all sit on the far side of the true edge for
    /// *some* variance, but only for one sign of `p + R − 1` each — one parameter set cannot catch
    /// them all.
    #[test]
    fn the_huber_weight_leaves_one_exactly_at_its_tuning_constant() {
        for (prior, noise, tuning) in [
            (4.0, 1.0, 1.345),
            (0.25, 4.0, 1.0),
            (10.0, 0.1, 2.5),
            (0.1, 0.2, 0.5),
            (0.05, 0.4, 0.8),
        ] {
            let boundary = tuning * libm::sqrt(prior + noise);
            for scale in [0.5, 0.999] {
                let w = huber_weight(prior, noise, scale * boundary, tuning);
                assert_eq!(
                    w, 1.0,
                    "p = {prior}, R = {noise}, c = {tuning} at {scale}× the boundary"
                );
            }
            for scale in [1.001, 1.5, 20.0] {
                let innovation = scale * boundary;
                let w = huber_weight(prior, noise, innovation, tuning);
                assert!(
                    w > 0.0 && w < 1.0,
                    "p = {prior}, R = {noise}, c = {tuning} at {scale}× the boundary: w = {w}"
                );
                let c2 = tuning * tuning;
                let residual = w * w * w * innovation * innovation - c2 * prior * w - c2 * noise;
                assert!(
                    residual.abs() < 1e-12 * c2 * (prior + noise),
                    "w³ν² − c²pw − c²R = {residual} at w = {w}"
                );
            }
        }
    }

    /// A prior with no uncertainty left in it, which is the one input where the fixed point has a
    /// closed form to check against.
    ///
    /// `h P⁻ hᵀ = 0` is the boundary of the physical range, not outside it — a perfectly known state
    /// projected onto a measurement — so every policy has to admit it. It also collapses the cubic to
    /// `w³ν² = c²R`, whose root is a cube root, so the bisection can be checked against arithmetic
    /// that does not bisect.
    ///
    /// Two assertions, because neither does the other's job. The cube root is the physics and it
    /// agrees to within an ulp — `libm::cbrt` and this bisection disagree in the last bit, and which
    /// of them is the correctly rounded one is not what is under test here. The pinned literal is the
    /// arithmetic: 60 halvings put the bracket below f64 resolution, so which of the two neighbouring
    /// doubles comes back is decided entirely by which side a midpoint landing exactly on the root is
    /// assigned to, and no tolerance can see that.
    #[test]
    fn with_no_prior_uncertainty_the_huber_weight_is_a_cube_root() {
        for (noise, tuning, innovation, pinned) in [
            (0.25, 1.0, 5.0, 0.215_443_469_003_188_34),
            (1.0, 2.0, 10.0, 0.341_995_189_335_339_34),
            (4.0, 1.0, 20.0, 0.215_443_469_003_188_34),
        ] {
            let w = huber_weight(0.0, noise, innovation, tuning);
            let closed_form = libm::cbrt(tuning * tuning * noise / (innovation * innovation));
            assert!(
                (w / closed_form - 1.0).abs() < 1e-15,
                "R = {noise}, c = {tuning}, ν = {innovation}: {w} against ∛(c²R/ν²) = {closed_form}"
            );
            assert_eq!(w, pinned, "R = {noise}, c = {tuning}, ν = {innovation}");
        }
        for policy in [
            RobustPolicy::Plain,
            RobustPolicy::ChiSquareGate { threshold: 9.0 },
            RobustPolicy::ChiSquareInflation { threshold: 9.0 },
            RobustPolicy::Huber { tuning: 1.345 },
            RobustPolicy::StudentT {
                degrees_of_freedom: 5.0,
            },
        ] {
            let effective = policy
                .effective_variance(0.0, 2.0, 1.0)
                .expect("a certain prior is a legal prior");
            assert!(effective > 0.0 && effective.is_finite(), "{policy:?}");
        }
    }

    #[test]
    fn the_student_t_variance_satisfies_its_own_defining_equation() {
        let (prior, noise, dof) = (4.0, 1.0, 5.0);
        for innovation in [0.0, 1.0, 5.0, 50.0, 1e4] {
            let effective = RobustPolicy::StudentT {
                degrees_of_freedom: dof,
            }
            .effective_variance(prior, noise, innovation)
            .unwrap();
            let lambda = noise / effective;
            let f = prior + effective;
            let expected = (dof + 1.0) / (dof + innovation * innovation / f);
            assert!(
                (lambda / expected - 1.0).abs() < 1e-12,
                "ν = {innovation}: λ = {lambda}, (n+1)/(n+δ²) = {expected}"
            );
        }
    }

    /// The influence function, measured rather than asserted: how far one observation can move the
    /// state as it gets worse without limit.
    ///
    /// The state increment is `K ν = p ν / (p + R_eff)`. Sweeping `ν` over five decades:
    ///
    /// * the gate rejects past its threshold, so the increment is exactly 0;
    /// * the Student-t saturates — `R_eff` grows as `|ν|` and the two cancel;
    /// * Huber does **not**, because its `R_eff` grows only as `|ν|^(2/3)`; the increment rises as
    ///   `|ν|^(1/3)`, so at `ν = 10⁵` a single observation still moves the state by a factor of 30
    ///   more than at `ν = 10²`.
    ///
    /// That last line is the reason the module documentation says to prefer the Student-t. It is a
    /// property of the estimator, not of this implementation, and it is worth having a test that
    /// would notice if someone "fixed" it by clipping.
    #[test]
    fn only_the_student_t_has_a_bounded_influence_function() {
        let (prior, noise) = (4.0, 1.0);
        let increment = |policy: RobustPolicy, innovation: f64| {
            policy
                .effective_variance(prior, noise, innovation)
                .map_or(0.0, |effective| prior * innovation / (prior + effective))
        };

        let gate = RobustPolicy::ChiSquareGate { threshold: 9.0 };
        assert_eq!(increment(gate, 1e3), 0.0);

        let t = RobustPolicy::StudentT {
            degrees_of_freedom: 5.0,
        };
        let saturated = increment(t, 1e5);
        assert!(
            (increment(t, 1e7) / saturated - 1.0).abs() < 1e-3,
            "Student-t saturates: {} then {}",
            saturated,
            increment(t, 1e7)
        );
        // And the displacement it saturates at is closed-form: `f → |ν|√(R/(n+1))` makes the
        // increment `p·ν/f → p√((n+1)/R)` = 4√6. This is the bound, not an observation about 1e5.
        let bound = prior * libm::sqrt(6.0 / noise);
        assert!(
            (saturated / bound - 1.0).abs() < 1e-4,
            "{saturated} against p√((n+1)/R) = {bound}"
        );

        // Inflation is redescending — `R_eff ∝ ν²` drives the increment back down as `1/|ν|`, so a
        // gross outlier moves the state *less* than a moderate one. Its worst case is at the
        // threshold, not at infinity.
        let inflate = RobustPolicy::ChiSquareInflation { threshold: 9.0 };
        let at_threshold = increment(inflate, 3.0 * libm::sqrt(prior + noise));
        assert!(
            increment(inflate, 1e3) < at_threshold / 20.0,
            "{at_threshold} then {}",
            increment(inflate, 1e3)
        );
        assert!(
            increment(inflate, 1e6) < increment(inflate, 1e3),
            "not descending"
        );
        // The descending arm is closed-form too: `R_eff = Rν²/(t(p+R))` makes the increment
        // `p·t·(p+R)/(R|ν|)` once `R_eff` dominates `p`.
        for nu in [1e3, 1e6] {
            let asymptote = prior * 9.0 * (prior + noise) / (noise * nu);
            assert!(
                (increment(inflate, nu) / asymptote - 1.0).abs() < 1e-3,
                "ν = {nu}: {} against {asymptote}",
                increment(inflate, nu)
            );
        }
        // Continuous at the threshold, unlike the gate: the two branches agree where they meet.
        let epsilon = 3.0 * libm::sqrt(prior + noise) * (1.0 + 1e-9);
        assert!((increment(inflate, epsilon) / at_threshold - 1.0).abs() < 1e-6);

        let huber = RobustPolicy::Huber { tuning: 1.345 };
        let (near, far) = (increment(huber, 1e2), increment(huber, 1e5));
        assert!(far > 10.0 * near, "Huber's influence grows: {near} → {far}");
        assert!(
            far > 2.0 * bound,
            "and past the Student-t's ceiling: {far} > {bound}"
        );
        // The cube-root law, once `p` is negligible against `R_eff` and the asymptote is clean:
        // 1000× the innovation is 10× the displacement, to 1 %. At `ν = 1e2` it is not yet — `p = 4`
        // against `R_eff = 17.7` is a fifth of the denominator — which is why the pair above is
        // 11.5× rather than 10×.
        let (deep, deeper) = (increment(huber, 1e6), increment(huber, 1e9));
        assert!(
            (deeper / deep / 10.0 - 1.0).abs() < 0.01,
            "|ν|^(1/3): {deep} → {deeper}"
        );
    }

    #[test]
    fn a_gated_observation_leaves_the_filter_exactly_as_it_was() {
        let mut filter = SquareRootKalman::new([0.0, 0.0], [[2.0, 0.0], [0.0, 1.0]]);
        let before = (filter.x, filter.covariance_sqrt());
        let outcome = robust_update(
            &mut filter,
            RobustPolicy::ChiSquareGate { threshold: 9.0 },
            &[1.0, 0.0],
            100.0,
            1.0,
        );
        assert!(!outcome.accepted);
        assert_eq!(outcome.weight, 0.0);
        assert_eq!(outcome.innovation_variance, 0.0);
        assert_eq!(before, (filter.x, filter.covariance_sqrt()));
        assert!(outcome.nominal_normalized_squared > 9.0);
    }

    /// The reported residual is the observation's, not the update's: the same number on the branch
    /// that applies the correction and on the branch that refuses it.
    ///
    /// `ν²/(h P⁻ hᵀ + R)` is what a caller tallies to decide whether the filter is consistent, so it
    /// has to be the nominal residual — measured against the variance the filter believed *before* any
    /// policy inflated it — and it has to survive a rejection, which is exactly the case a
    /// consistency check is being run to find. The expectation here is formed from `P` rather than
    /// from the factor, so the two agree only if both routes to `h P⁻ hᵀ` do.
    #[test]
    fn the_nominal_residual_is_reported_on_both_branches() {
        let seed = SquareRootKalman::new([0.0, 0.0], [[2.0, 0.0], [0.5, 1.0]]);
        let h = [1.0, -0.5];
        let (innovation, noise) = (1.5, 1.0);

        let p = seed.covariance();
        let prior: f64 = (0..2)
            .map(|i| h[i] * (0..2).map(|j| p[i][j] * h[j]).sum::<f64>())
            .sum();
        assert_eq!(prior, 3.3125, "the geometry this test is pinned to moved");
        let expected = innovation * innovation / (prior + noise);

        let mut applied = seed.clone();
        let accepted = robust_update(&mut applied, RobustPolicy::Plain, &h, innovation, noise);
        assert!(accepted.accepted);
        assert_eq!(accepted.nominal_normalized_squared, expected);

        // A gate tight enough to refuse it reports the identical number, and reports it against the
        // nominal `R` rather than against the variance the gate never got to use.
        let mut refused = seed;
        let outcome = robust_update(
            &mut refused,
            RobustPolicy::ChiSquareGate { threshold: 1e-6 },
            &h,
            innovation,
            noise,
        );
        assert!(!outcome.accepted);
        assert_eq!(outcome.nominal_normalized_squared, expected);
    }

    /// An observation carrying no information at all still reports a residual, and reports it as
    /// infinite rather than as a NaN.
    ///
    /// `h = 0` makes the projected prior exactly zero, and a zero `R` is refused by every policy, so
    /// this is the one input for which the nominal variance vanishes and the normalised residual has
    /// no value. `f64::INFINITY` says "arbitrarily far outside", which is what a caller screening on
    /// it should conclude; `0/0 = NaN` compares false against every threshold and would pass a
    /// consistency check by saying nothing.
    #[test]
    fn a_measurement_with_no_variance_at_all_reports_an_infinite_residual() {
        let mut filter = SquareRootKalman::new([0.0, 0.0], [[2.0, 0.0], [0.5, 1.0]]);
        let before = (filter.x, filter.covariance_sqrt());
        let outcome = robust_update(&mut filter, RobustPolicy::Plain, &[0.0, 0.0], 0.0, 0.0);
        assert!(!outcome.accepted);
        assert!(
            outcome.nominal_normalized_squared.is_infinite(),
            "{}",
            outcome.nominal_normalized_squared
        );
        assert_eq!(before, (filter.x, filter.covariance_sqrt()));
    }

    #[test]
    fn the_plain_policy_is_the_plain_update() {
        let seed = SquareRootKalman::new([0.5, -0.25], [[2.0, 0.0], [0.3, 1.0]]);
        let (mut robust, mut plain) = (seed.clone(), seed);
        let outcome = robust_update(&mut robust, RobustPolicy::Plain, &[1.0, 0.5], 0.7, 1.5);
        let variance = plain.measurement_update(&[1.0, 0.5], 0.7, 1.5);
        assert_eq!(robust.x, plain.x);
        assert_eq!(robust.covariance_sqrt(), plain.covariance_sqrt());
        assert_eq!(outcome.innovation_variance, variance);
        assert_eq!(outcome.weight, 1.0);
    }

    #[test]
    fn the_tally_counts_what_the_filter_did() {
        let mut tally = RobustTally::default();
        let outcome = |accepted, weight| RobustOutcome {
            accepted,
            weight,
            nominal_normalized_squared: 0.0,
            innovation_variance: 1.0,
        };
        tally.record(&outcome(true, 1.0));
        tally.record(&outcome(true, 0.5));
        tally.record(&outcome(false, 0.0));
        tally.record(&outcome(true, 1.0));
        assert_eq!(tally.seen, 4);
        assert_eq!(tally.rejected, 1);
        assert_eq!(tally.downweighted, 1);
        assert_eq!(tally.mean_weight(), 2.5 / 4.0);
        assert_eq!(tally.longest_rejection_run, 1);
        assert_eq!(tally.rejected_fraction(), 0.25);
        // Runs are broken by an acceptance, and only the longest survives.
        let mut runs = RobustTally::default();
        for accepted in [false, false, true, false, false, false, true, false] {
            runs.record(&outcome(accepted, 1.0));
        }
        assert_eq!(runs.longest_rejection_run, 3);
        assert_eq!(runs.rejected, 6);
        assert_eq!(RobustTally::default().longest_rejection_run, 0);
        assert_eq!(RobustTally::default().mean_weight(), 0.0);
        assert_eq!(RobustTally::default().rejected_fraction(), 0.0);
    }

    #[test]
    fn an_unposeable_update_is_refused_rather_than_guessed_at() {
        for policy in [
            RobustPolicy::Plain,
            RobustPolicy::ChiSquareGate { threshold: 9.0 },
            RobustPolicy::ChiSquareInflation { threshold: 9.0 },
            RobustPolicy::Huber { tuning: 1.345 },
            RobustPolicy::StudentT {
                degrees_of_freedom: 5.0,
            },
        ] {
            assert_eq!(policy.effective_variance(1.0, 0.0, 1.0), None);
            assert_eq!(policy.effective_variance(1.0, -1.0, 1.0), None);
            assert_eq!(policy.effective_variance(-1.0, 1.0, 1.0), None);
            assert_eq!(policy.effective_variance(1.0, 1.0, f64::NAN), None);
            assert_eq!(policy.effective_variance(f64::INFINITY, 1.0, 1.0), None);
        }
        assert_eq!(
            RobustPolicy::Huber { tuning: 0.0 }.effective_variance(1.0, 1.0, 1.0),
            None
        );
        assert_eq!(
            RobustPolicy::ChiSquareInflation { threshold: 0.0 }.effective_variance(1.0, 1.0, 1.0),
            None
        );
        assert_eq!(
            RobustPolicy::StudentT {
                degrees_of_freedom: 0.0
            }
            .effective_variance(1.0, 1.0, 1.0),
            None
        );
    }

    /// At one degree of freedom the three *threshold* policies must reproduce the scalar path exactly
    /// — a one-dimensional "vector" observation is the scalar case, and two answers to the same
    /// question is how a second code path starts drifting from the first.
    ///
    /// Huber and the Student-t must not, and the difference is the whole content of
    /// [`variance_scale`](RobustPolicy::variance_scale)'s documentation: the scalar path solves a fixed
    /// point in which `δ` is measured against the *inflated* variance, and the vector path measures it
    /// once against the nominal one. Since the nominal `S` is the smaller of the two, the one-step form
    /// sees a larger standardised residual and therefore discounts **more**, not less — checked in
    /// [`the_one_step_forms_discount_harder_than_the_fixed_points_they_approximate`].
    #[test]
    fn one_degree_of_freedom_reproduces_the_scalar_threshold_policies() {
        // δ² = ν²/(p+R) with p = R = 0.5 gives ν² = δ², so the two entry points take the same argument.
        for delta_squared in [0.0, 0.25, 1.0, 4.0, 9.0, 100.0] {
            let nu = libm::sqrt(delta_squared);
            for policy in [
                RobustPolicy::Plain,
                RobustPolicy::ChiSquareGate { threshold: 9.0 },
                RobustPolicy::ChiSquareInflation { threshold: 9.0 },
            ] {
                let scalar = policy.effective_variance(0.5, 0.5, nu);
                let vector = policy.variance_scale(delta_squared, 1).map(|s| 0.5 * s);
                match (scalar, vector) {
                    (None, None) => {}
                    (Some(a), Some(b)) => assert!(
                        (a - b).abs() < 1e-12 * a.max(1.0),
                        "{policy:?} at δ² = {delta_squared}: scalar {a}, vector {b}"
                    ),
                    _ => {
                        panic!("{policy:?} at δ² = {delta_squared}: {scalar:?} against {vector:?}")
                    }
                }
            }
        }
    }

    /// The two M-estimating policies are one-step here and fixed points there, and the inequality goes
    /// one way: past the point where either discounts at all, the one-step form always discounts harder,
    /// because it standardises against a smaller `S`. Both meet the fixed point exactly at the boundary,
    /// where the weight is 1 either way.
    ///
    /// This is the error budget of the approximation, stated as a direction rather than as a bound: a
    /// caller porting a tuning constant from the scalar path to a vector filter gets a *more*
    /// aggressive estimator than the one they tuned, never a less aggressive one.
    #[test]
    fn the_one_step_forms_discount_harder_than_the_fixed_points_they_approximate() {
        for policy in [
            RobustPolicy::Huber { tuning: 1.345 },
            RobustPolicy::StudentT {
                degrees_of_freedom: 5.0,
            },
        ] {
            for delta_squared in [4.0, 9.0, 25.0, 100.0, 1e4] {
                let nu = libm::sqrt(delta_squared);
                let fixed = policy.effective_variance(0.5, 0.5, nu).unwrap();
                let one_step = 0.5 * policy.variance_scale(delta_squared, 1).unwrap();
                assert!(
                    one_step > fixed,
                    "{policy:?} at δ² = {delta_squared}: one-step R_eff {one_step} against fixed \
                     point {fixed}"
                );
            }
        }
        // At the tuning constant both Huber forms are exactly the nominal noise.
        let boundary = 1.345 * 1.345;
        assert_eq!(
            RobustPolicy::Huber { tuning: 1.345 }.variance_scale(boundary, 1),
            Some(1.0)
        );
        assert_eq!(
            RobustPolicy::Huber { tuning: 1.345 }.effective_variance(0.5, 0.5, 1.345),
            Some(0.5)
        );
    }

    /// The per-degree-of-freedom standardisation is the reason `c = 1.345` keeps its meaning as the
    /// dimension grows. `E[δ²] = d`, so an observation at its own expectation must be undiscounted at
    /// every `d` — otherwise a three-axis star tracker would be permanently down-weighted for being
    /// three-axis.
    #[test]
    fn a_typical_observation_is_undiscounted_at_every_dimension() {
        for dof in 1_u32..=6 {
            let expectation = f64::from(dof);
            let dof = dof as usize;
            assert_eq!(
                RobustPolicy::Huber { tuning: 1.345 }.variance_scale(expectation, dof),
                Some(1.0),
                "Huber discounts a typical {dof}-vector"
            );
            // The Student-t E-step is exactly 1 at δ² = d — that is what makes its scale the mean
            // effective noise rather than a number that has to be re-tuned per dimension.
            let student = RobustPolicy::StudentT {
                degrees_of_freedom: 5.0,
            }
            .variance_scale(expectation, dof)
            .unwrap();
            assert_eq!(student, 1.0, "Student-t at δ² = {dof} gave {student}");
        }
    }

    /// The gate applies its threshold to `δ²` **whole**, not per axis. Checked at the boundary rather
    /// than by sampling, because the boundary is where the two readings differ: a per-axis gate at the
    /// same number would admit `δ² = 3·threshold` (each axis at the threshold) and reject
    /// `δ² = threshold` split unevenly.
    #[test]
    fn the_gate_thresholds_the_whole_normalised_innovation_and_not_one_axis_of_it() {
        let threshold = chi_square_gate(3, 2.576);
        let policy = RobustPolicy::ChiSquareGate { threshold };
        assert_eq!(policy.variance_scale(threshold, 3), Some(1.0));
        assert_eq!(policy.variance_scale(threshold * (1.0 + 1e-12), 3), None);
        // Three axes each sitting exactly at the threshold is 3× the threshold in total, and is
        // rejected — the reading a per-axis gate would admit.
        assert_eq!(policy.variance_scale(3.0 * threshold, 3), None);
        // The threshold does not move with the dimension the caller declares; only the caller's choice
        // of `chi_square_gate(dof, z)` does. Passing a χ²(3) threshold with `dof = 1` is a caller
        // error, and it shows up as an unusually permissive gate rather than as a refusal.
        assert_eq!(policy.variance_scale(threshold, 1), Some(1.0));
        assert!(chi_square_gate(1, 2.576) < threshold);
    }

    #[test]
    fn a_scale_cannot_be_produced_for_an_input_that_is_not_one() {
        for policy in [
            RobustPolicy::Plain,
            RobustPolicy::ChiSquareInflation { threshold: 9.0 },
            RobustPolicy::Huber { tuning: 1.345 },
            RobustPolicy::StudentT {
                degrees_of_freedom: 5.0,
            },
        ] {
            assert_eq!(policy.variance_scale(f64::NAN, 3), None);
            assert_eq!(policy.variance_scale(f64::INFINITY, 3), None);
            assert_eq!(policy.variance_scale(-1.0, 3), None);
            // Zero degrees of freedom is not an observation, and dividing by it would give the Huber
            // branch an infinite radius rather than a refusal.
            assert_eq!(policy.variance_scale(1.0, 0), None);
        }
        // A policy whose own parameter is unusable is refused rather than silently treated as Plain.
        assert_eq!(
            RobustPolicy::Huber { tuning: 0.0 }.variance_scale(4.0, 3),
            None
        );
        assert_eq!(
            RobustPolicy::ChiSquareInflation { threshold: 0.0 }.variance_scale(4.0, 3),
            None
        );
        assert_eq!(
            RobustPolicy::StudentT {
                degrees_of_freedom: 0.0
            }
            .variance_scale(4.0, 3),
            None
        );
    }

    /// Both redescending policies must scale as `δ²` in the tail — that is what makes the state
    /// increment fall as `1/|ν|`, and it is the property `tests/robust_vector_estimation.rs` measures
    /// as the difference between 507″ and 660″ at 20 % contamination.
    #[test]
    fn inflation_and_the_student_t_are_asymptotically_quadratic_and_huber_is_not() {
        let (a, b) = (1e6, 4e6);
        for policy in [
            RobustPolicy::ChiSquareInflation { threshold: 9.0 },
            RobustPolicy::StudentT {
                degrees_of_freedom: 5.0,
            },
        ] {
            let ratio = policy.variance_scale(b, 3).unwrap() / policy.variance_scale(a, 3).unwrap();
            assert!((ratio - 4.0).abs() < 1e-3, "{policy:?} scaled by {ratio}");
        }
        let huber = RobustPolicy::Huber { tuning: 1.345 };
        let ratio = huber.variance_scale(b, 3).unwrap() / huber.variance_scale(a, 3).unwrap();
        assert!(
            (ratio - 2.0).abs() < 1e-9,
            "Huber scaled by {ratio}, not by √4"
        );
    }
}
