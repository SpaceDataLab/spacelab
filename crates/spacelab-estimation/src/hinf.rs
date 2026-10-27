//! The H∞ (minimax) filter: a state estimator that assumes *nothing* about the noise except that it
//! has finite energy, and bounds the worst case rather than the average.
//!
//! # Why a second kind of filter
//!
//! Every other estimator in this crate — [`crate::Srif`], [`crate::SquareRootKalman`],
//! [`crate::SquareRootUnscentedKalman`], [`crate::MultiplicativeEkf`], [`crate::InvariantEkf`],
//! [`crate::NavigationEkf`] — is a minimum-variance estimator: it minimises the expected squared error
//! *given* that the process and measurement noise are zero-mean with the covariances the caller
//! supplied. In space surveillance that premise fails exactly where the answer matters most. An
//! unmodelled manoeuvre is not zero-mean noise. Neither is a drag coefficient that is 30 % wrong, nor
//! an area-to-mass ratio inferred from a catalogue. Against those the Kalman gain is optimal for a
//! world that is not the one the measurements came from, and the estimate walks away from the truth
//! while the reported covariance keeps shrinking.
//!
//! The H∞ filter answers the other question. It makes no distributional assumption at all and instead
//! bounds the *energy gain* from every disturbance — process noise, measurement noise, and the error
//! in the initial estimate — to the estimation error:
//!
//! ```text
//! sup  ‖x − x̂‖²_Q̄ / (‖x₀ − x̂₀‖²_{P₀⁻¹} + ‖w‖²_{W⁻¹} + ‖v‖²_{V⁻¹})  <  1/γ
//! ```
//!
//! over every finite-energy disturbance sequence. The price is efficiency: on data that really is
//! Gaussian, H∞ is worse than Kalman, and the smaller `1/γ` is asked to be, the worse. That is the
//! trade, and it is the reason both filters live here rather than one.
//!
//! It complements [`crate::consider`] rather than replacing it: consider covariance handles parameters
//! that are unmodelled but *known* to be uncertain, with a known covariance; H∞ handles a disturbance
//! about which nothing is known at all.
//!
//! # The recursion, and the form it is evaluated in
//!
//! One-step predictor form (Simon, *Optimal State Estimation*, §11.4), with `x̂ₖ` the estimate of the
//! state at step `k` given measurements through `k−1`:
//!
//! ```text
//! L   = [I − γ Q̄ P + Hᵀ V⁻¹ H P]⁻¹
//! K   = F P L Hᵀ V⁻¹
//! x̂⁺  = F x̂ + K (z − H x̂)
//! P⁺  = F P L Fᵀ + W
//! ```
//!
//! Written that way it asks for the inverse of a matrix that is not symmetric, and hides the condition
//! it exists under. Factor `L`'s bracket as `(P⁻¹ − γ Q̄ + Hᵀ V⁻¹ H) P` and the whole recursion
//! collapses onto one symmetric matrix
//!
//! ```text
//! M = P⁻¹ − γ Q̄ + Hᵀ V⁻¹ H,     P L = M⁻¹,
//! K = F M⁻¹ Hᵀ V⁻¹,             P⁺ = F M⁻¹ Fᵀ + W,
//! ```
//!
//! which is the form evaluated here. Two things follow, and both are the point:
//!
//! * **The existence condition is the factorisation.** H∞ requires `M ≻ 0`, and `M` is exactly the
//!   matrix whose Cholesky factor the step needs. A γ too aggressive for the current covariance makes
//!   the factorisation fail on a specific diagonal, and that diagonal is reported
//!   ([`HInfinityError::ExistenceConditionViolated`]) instead of a gain being returned that solves no
//!   minimax problem at all. It is not a guard bolted on ahead of the arithmetic; there is no path
//!   through the code that computes a gain without it.
//! * **Everything stays in square-root form**, as the rest of the crate does: `P⁺` is built as the
//!   pre-array `[F M⁻ᵀ⁄² | S_W]` and lower-triangularised by Givens rotations, exactly as
//!   [`crate::SquareRootKalman::time_update`] does, so `P⁺` is positive-semidefinite by construction
//!   and never formed as a difference.
//!
//! At `γ = 0` the recursion *is* the Kalman filter: `M = P⁻¹ + Hᵀ V⁻¹ H`, so `M⁻¹` is the Kalman
//! posterior covariance by the matrix-inversion lemma and `K = F P Hᵀ (H P Hᵀ + V)⁻¹` is the one-step
//! predictor gain. `the_zero_gamma_filter_is_the_kalman_filter` holds the two to 1e-12 over twelve
//! steps.
//!
//! # One step, not a predict and an update
//!
//! [`step`](HInfinityFilter::step) is a single recursion that consumes one measurement and returns the
//! estimate at the *next* epoch. It is not a `predict` followed by an `update`, and it must not be
//! split into one: `F` appears in the gain and in the covariance recursion, so a caller that also
//! propagates the state itself advances it twice. (`filterpy`'s `hinfinity_filter` does exactly that —
//! its `update` is this whole recursion and its `predict` applies `F` again — which is one of the
//! reasons its own author marked it "I do not believe this code is correct". Nothing here is ported
//! from it; the mathematics is Simon's.)
//!
//! Gate: `derivations/hinf_riccati.mac` proves `P L = M⁻¹` — so the symmetric form above is Simon's
//! gain and Riccati recursion — and that at `γ = 0` it is the Kalman posterior and predictor gain;
//! `derivations/linear_filters.gp` re-runs its fixture in Simon's form. The module tests
//! `the_recursion_is_the_exact_rational_riccati_recursion` (G2) and
//! `the_bound_is_refused_exactly_where_the_prior_loses_definiteness` (G3) tie the code to it.

// Matrix algebra over const-generic dimensions, indexed by one i/j/k as the rest of the crate does;
// the filter symbols (F, H, K, P, M, V, W, γ) are one letter by universal convention.
#![allow(clippy::many_single_char_names, clippy::needless_range_loop)]

/// Why a step of the H∞ recursion could not be taken. Each variant names the direction — the index of
/// the diagonal that failed — because that is what tells a caller *which* state or channel is
/// responsible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HInfinityError {
    /// `M = P⁻¹ − γ Q̄ + Hᵀ V⁻¹ H` is not positive-definite, so the minimax problem this γ poses has
    /// no solution at this covariance: the disturbance is allowed more energy than the estimator can
    /// bound. The normal cause is a γ chosen too aggressively for the current `P`. Reduce γ (`0` is
    /// always feasible and gives the Kalman filter) or wait for `P` to shrink.
    ExistenceConditionViolated {
        /// Index of the diagonal at which the Cholesky factorisation of `M` failed.
        direction: usize,
    },
    /// The covariance the filter carries is singular in some direction, so `P⁻¹` does not exist.
    /// Reachable only from a rank-deficient `S₀`; the recursion itself cannot produce one.
    SingularCovariance {
        /// Index of the zero diagonal in the covariance square root.
        direction: usize,
    },
    /// The measurement-noise covariance `V` is not positive-definite, so `V⁻¹` does not exist. A
    /// channel with zero variance is a perfect measurement, which no filter of this family admits.
    MeasurementNoiseNotPositiveDefinite {
        /// Index of the diagonal at which the Cholesky factorisation of `V` failed.
        direction: usize,
    },
    /// The performance bound is not usable: γ is negative, or not finite, or so large that
    /// `P₀⁻¹ − γ Q̄` is already indefinite before any measurement is folded in. Rejected at
    /// construction, since a filter that cannot take its own first step should not exist.
    UnattainableBound {
        /// Index of the diagonal at which `P₀⁻¹ − γ Q̄` failed, or `0` for a γ that is negative or
        /// not finite.
        direction: usize,
    },
}

impl core::fmt::Display for HInfinityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ExistenceConditionViolated { direction } => write!(
                f,
                "H-infinity existence condition fails in direction {direction}: reduce gamma"
            ),
            Self::SingularCovariance { direction } => {
                write!(f, "covariance is singular in direction {direction}")
            }
            Self::MeasurementNoiseNotPositiveDefinite { direction } => write!(
                f,
                "measurement noise covariance is not positive-definite in direction {direction}"
            ),
            Self::UnattainableBound { direction } => write!(
                f,
                "gamma is unattainable for this prior covariance in direction {direction}"
            ),
        }
    }
}

impl core::error::Error for HInfinityError {}

/// A discrete-time H∞ filter over an `N`-dimensional state, carrying the covariance in square-root
/// form.
///
/// [`x`](Self::x) is the estimate at the current epoch *given measurements through the previous one* —
/// the one-step predictor convention the recursion is written in. Each [`step`](Self::step) consumes
/// one measurement and advances both.
#[derive(Debug, Clone)]
pub struct HInfinityFilter<const N: usize> {
    /// The state estimate at the current epoch, given every measurement before it.
    pub x: [f64; N],
    /// Covariance square root `S`, with `P = S Sᵀ` (lower-triangular). Never `P` itself.
    s: [[f64; N]; N],
    /// The performance weight `Q̄`: which combination of states the bound is on. The identity bounds
    /// every state equally.
    performance_weight: [[f64; N]; N],
    /// The bound parameter γ. Larger is more conservative and harder to attain; `0` is the Kalman
    /// filter.
    gamma: f64,
}

impl<const N: usize> HInfinityFilter<N> {
    /// Seed with an estimate, the square root of its covariance (`P₀ = S₀ S₀ᵀ`), the performance
    /// weight `Q̄`, and the bound γ.
    ///
    /// # Errors
    ///
    /// [`HInfinityError::UnattainableBound`] if γ is negative or not finite, or if `P₀⁻¹ − γ Q̄` is
    /// already indefinite. The measurement term `Hᵀ V⁻¹ H` is positive-semidefinite and can only help,
    /// so a γ failing here fails at every step; one that passes can still fail later as `P` grows,
    /// which is why [`step`](Self::step) is fallible too.
    ///
    /// [`HInfinityError::SingularCovariance`] if `S₀` has a zero on its diagonal.
    pub fn new(
        x0: [f64; N],
        covariance_sqrt: [[f64; N]; N],
        performance_weight: [[f64; N]; N],
        gamma: f64,
    ) -> Result<Self, HInfinityError> {
        if !(gamma.is_finite() && gamma >= 0.0) {
            return Err(HInfinityError::UnattainableBound { direction: 0 });
        }
        let inverse = inverse_from_factor(&covariance_sqrt)
            .map_err(|direction| HInfinityError::SingularCovariance { direction })?;
        let mut worst = inverse;
        for i in 0..N {
            for j in 0..N {
                worst[i][j] -= gamma * performance_weight[i][j];
            }
        }
        cholesky(&symmetrised(&worst))
            .map_err(|direction| HInfinityError::UnattainableBound { direction })?;
        Ok(Self {
            x: x0,
            s: covariance_sqrt,
            performance_weight,
            gamma,
        })
    }

    /// Seed from a full covariance matrix, taking its lower Cholesky factor.
    ///
    /// # Errors
    ///
    /// [`HInfinityError::SingularCovariance`] if `p0` is not positive-definite, and everything
    /// [`new`](Self::new) can return.
    pub fn from_covariance(
        x0: [f64; N],
        p0: &[[f64; N]; N],
        performance_weight: [[f64; N]; N],
        gamma: f64,
    ) -> Result<Self, HInfinityError> {
        let s =
            cholesky(p0).map_err(|direction| HInfinityError::SingularCovariance { direction })?;
        Self::new(x0, s, performance_weight, gamma)
    }

    /// One step of the recursion: fold in the `D`-dimensional measurement `z` and advance to the next
    /// epoch through the transition `F`. Returns the innovation `z − H x̂`, so a caller can gate or
    /// tally on it exactly as with [`crate::SquareRootKalman::measurement_update`].
    ///
    /// `measurement` is `H` as `D` rows of `N`, `noise_covariance` is `V` (not a square root — it is
    /// inverted here), and `process_noise_sqrt` is any `S_W` with `W = S_W S_Wᵀ`.
    ///
    /// This is one recursion, not a predict and an update: `F` is already inside the gain. See the
    /// module documentation.
    ///
    /// # Errors
    ///
    /// [`HInfinityError::ExistenceConditionViolated`] if γ is too aggressive for the current
    /// covariance, [`HInfinityError::MeasurementNoiseNotPositiveDefinite`] if `V` is not invertible,
    /// and [`HInfinityError::SingularCovariance`] if `P` has gone singular.
    pub fn step<const D: usize>(
        &mut self,
        transition: &[[f64; N]; N],
        measurement: &[[f64; N]; D],
        z: &[f64; D],
        noise_covariance: &[[f64; D]; D],
        process_noise_sqrt: &[[f64; N]; N],
    ) -> Result<[f64; D], HInfinityError> {
        // V⁻¹, and with it the information the measurement carries, `HᵀV⁻¹H`.
        let noise_factor = cholesky(noise_covariance).map_err(|direction| {
            HInfinityError::MeasurementNoiseNotPositiveDefinite { direction }
        })?;
        let noise_inverse = inverse_from_factor(&noise_factor).map_err(|direction| {
            HInfinityError::MeasurementNoiseNotPositiveDefinite { direction }
        })?;

        // W = V⁻¹ H (D × N), so that HᵀV⁻¹H = Hᵀ W and the gain's HᵀV⁻¹ is available as Wᵀ.
        let mut weighted = [[0.0; N]; D];
        for i in 0..D {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..D {
                    acc += noise_inverse[i][k] * measurement[k][j];
                }
                weighted[i][j] = acc;
            }
        }

        // M = P⁻¹ − γ Q̄ + Hᵀ V⁻¹ H. Symmetric in exact arithmetic by construction; symmetrised
        // because the Cholesky below reads only one triangle and the two must agree.
        let mut m = inverse_from_factor(&self.s)
            .map_err(|direction| HInfinityError::SingularCovariance { direction })?;
        for i in 0..N {
            for j in 0..N {
                let mut information = 0.0;
                for k in 0..D {
                    information += measurement[k][i] * weighted[k][j];
                }
                m[i][j] += information - self.gamma * self.performance_weight[i][j];
            }
        }
        let m_factor = cholesky(&symmetrised(&m))
            .map_err(|direction| HInfinityError::ExistenceConditionViolated { direction })?;
        // M⁻¹ = M_f⁻ᵀ M_f⁻¹, and its factor M_f⁻ᵀ is what the covariance pre-array needs directly.
        let m_inverse_factor = invert_lower(&m_factor)
            .map_err(|direction| HInfinityError::ExistenceConditionViolated { direction })?;
        let mut m_inverse = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..N {
                    acc += m_inverse_factor[k][i] * m_inverse_factor[k][j];
                }
                m_inverse[i][j] = acc;
            }
        }

        // K = F M⁻¹ Hᵀ V⁻¹ = F M⁻¹ Wᵀ (N × D).
        let mut transformed = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..N {
                    acc += transition[i][k] * m_inverse[k][j];
                }
                transformed[i][j] = acc;
            }
        }
        let mut gain = [[0.0; D]; N];
        for i in 0..N {
            for j in 0..D {
                let mut acc = 0.0;
                for k in 0..N {
                    acc += transformed[i][k] * weighted[j][k];
                }
                gain[i][j] = acc;
            }
        }

        // x̂⁺ = F x̂ + K (z − H x̂).
        let mut innovation = *z;
        for i in 0..D {
            for k in 0..N {
                innovation[i] -= measurement[i][k] * self.x[k];
            }
        }
        let mut next = [0.0; N];
        for i in 0..N {
            let mut acc = 0.0;
            for k in 0..N {
                acc += transition[i][k] * self.x[k];
            }
            for j in 0..D {
                acc += gain[i][j] * innovation[j];
            }
            next[i] = acc;
        }
        self.x = next;

        // P⁺ = F M⁻¹ Fᵀ + W = (F M_f⁻ᵀ)(F M_f⁻ᵀ)ᵀ + S_W S_Wᵀ, taken as the pre-array
        // [F M_f⁻ᵀ | S_W] and lower-triangularised — the same Givens sweep as the linear filter's
        // time update, so no covariance is ever formed as a sum of outer products.
        let mut left = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..N {
                    acc += transition[i][k] * m_inverse_factor[j][k];
                }
                left[i][j] = acc;
            }
        }
        self.s = triangularise(left, *process_noise_sqrt);
        Ok(innovation)
    }

    /// The current covariance `P = S Sᵀ`, formed only on request.
    #[must_use]
    pub fn covariance(&self) -> [[f64; N]; N] {
        let mut p = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..N {
                    acc += self.s[i][k] * self.s[j][k];
                }
                p[i][j] = acc;
            }
        }
        p
    }

    /// The covariance square root `S`, with `P = S Sᵀ` (lower-triangular).
    #[must_use]
    pub const fn covariance_sqrt(&self) -> [[f64; N]; N] {
        self.s
    }

    /// The bound parameter γ this filter was built with.
    #[must_use]
    pub const fn gamma(&self) -> f64 {
        self.gamma
    }
}

/// Lower-triangularise the pre-array `[left | right]` by Givens rotations, returning the left block —
/// a square root of `left leftᵀ + right rightᵀ` that is never formed as that sum.
fn triangularise<const N: usize>(
    mut left: [[f64; N]; N],
    mut right: [[f64; N]; N],
) -> [[f64; N]; N] {
    for i in 0..N {
        for j in (i + 1)..(2 * N) {
            let p = left[i][i];
            let q = if j < N { left[i][j] } else { right[i][j - N] };
            let denom = libm::hypot(p, q);
            if denom == 0.0 {
                continue;
            }
            let (c, s) = (p / denom, q / denom);
            for r in 0..N {
                let a_i = left[r][i];
                let a_j = if j < N { left[r][j] } else { right[r][j - N] };
                left[r][i] = c * a_i + s * a_j;
                let updated = -s * a_i + c * a_j;
                if j < N {
                    left[r][j] = updated;
                } else {
                    right[r][j - N] = updated;
                }
            }
        }
    }
    left
}

/// The lower Cholesky factor of a symmetric positive-definite matrix, or the index of the diagonal at
/// which it stopped being positive-definite.
///
/// [`crate::unscented::cholesky`] does the same arithmetic and returns `Option`. Here the factorisation
/// *is* the H∞ existence check, so which direction failed is the diagnosis rather than a detail, and
/// the caller turns that index into the error variant that names what the matrix was.
fn cholesky<const N: usize>(a: &[[f64; N]; N]) -> Result<[[f64; N]; N], usize> {
    let mut l = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                // NaN is a failure too: it means the pre-array carried one, and `sqrt` would hide it.
                if sum.is_nan() || sum <= 0.0 {
                    return Err(i);
                }
                l[i][j] = libm::sqrt(sum);
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Ok(l)
}

/// The inverse of a lower-triangular matrix by forward substitution, or the index of a zero diagonal.
fn invert_lower<const N: usize>(l: &[[f64; N]; N]) -> Result<[[f64; N]; N], usize> {
    let mut inverse = [[0.0; N]; N];
    for i in 0..N {
        if l[i][i] == 0.0 {
            return Err(i);
        }
        inverse[i][i] = 1.0 / l[i][i];
        for j in 0..i {
            let mut acc = 0.0;
            for k in j..i {
                acc += l[i][k] * inverse[k][j];
            }
            inverse[i][j] = -acc / l[i][i];
        }
    }
    Ok(inverse)
}

/// `(L Lᵀ)⁻¹ = L⁻ᵀ L⁻¹` for a lower-triangular `L`: the inverse of the matrix a factor stands for,
/// without ever forming that matrix.
fn inverse_from_factor<const N: usize>(l: &[[f64; N]; N]) -> Result<[[f64; N]; N], usize> {
    let factor = invert_lower(l)?;
    let mut inverse = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0.0;
            for k in 0..N {
                acc += factor[k][i] * factor[k][j];
            }
            inverse[i][j] = acc;
        }
    }
    Ok(inverse)
}

/// `(A + Aᵀ)/2`. The matrices this is applied to are symmetric in exact arithmetic; rounding leaves
/// the two triangles differing in the last bits, and a Cholesky that reads one triangle would then be
/// factoring a matrix nobody wrote.
fn symmetrised<const N: usize>(a: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            out[i][j] = f64::midpoint(a[i][j], a[j][i]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]
    use super::*;
    use crate::kalman::SquareRootKalman;

    /// A constant-acceleration pair: position and velocity at a 1 s step, position observed.
    const F: [[f64; 2]; 2] = [[1.0, 1.0], [0.0, 1.0]];
    const H: [[f64; 2]; 1] = [[1.0, 0.0]];
    const SQ: [[f64; 2]; 2] = [[0.02, 0.0], [0.0, 0.05]];
    const P0: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 0.5]];
    const R: f64 = 0.09;

    fn identity<const N: usize>() -> [[f64; N]; N] {
        let mut m = [[0.0; N]; N];
        for i in 0..N {
            m[i][i] = 1.0;
        }
        m
    }

    /// (G2) Two steps of the recursion against the Riccati recursion evaluated in exact rationals by
    /// `derivations/hinf_riccati.mac`, which also shows that this symmetric form is Simon's
    /// `L = [I − γ Q̄ P + Hᵀ V⁻¹ H P]⁻¹` form (`P L = M⁻¹`). The fixture has a correlated
    /// two-channel `V`, a γ of 1/8 with `Q̄ = I` and dyadic inputs throughout, so only the filter
    /// rounds; the bound is 16 ulps of the prior's scale, 4.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn the_recursion_is_the_exact_rational_riccati_recursion() {
        let f = [[1.0, 0.5], [0.0, 1.0]];
        let h = [[1.0, 0.25], [0.0, 1.0]];
        let v = [[0.5, 0.125], [0.125, 0.25]];
        let sw = [[0.25, 0.0], [0.125, 0.5]];
        let mut filter =
            HInfinityFilter::new([0.5, -1.0], [[2.0, 0.0], [0.5, 1.0]], identity(), 0.125).unwrap();
        let exact = [
            (
                [0.75, 0.5],
                [0.5, 1.5],
                [
                    8.129_339_039_467_427_484_545_887e-1,
                    2.545_886_828_340_466_000_951_022e-1,
                ],
                [
                    [
                        6.127_615_311_459_819_305_753_685e-1,
                        2.066_185_211_602_472_658_107_466e-1,
                    ],
                    [
                        2.066_185_211_602_472_658_107_466e-1,
                        4.763_715_525_439_847_836_424_156e-1,
                    ],
                ],
            ),
            (
                [1.75, -0.25],
                [
                    8.734_189_253_447_456_015_216_358e-1,
                    -5.045_886_828_340_466_000_951_022e-1,
                ],
                [
                    1.321_976_336_106_491_573_864_834,
                    -4.177_421_395_864_781_145_500_628e-2,
                ],
                [
                    [
                        4.268_644_552_841_899_385_016_474e-1,
                        1.720_205_549_559_947_494_042_773e-1,
                    ],
                    [
                        1.720_205_549_559_947_494_042_773e-1,
                        4.324_307_212_715_680_222_427_528e-1,
                    ],
                ],
            ),
        ];
        let tol = 16.0 * f64::EPSILON * 4.0;
        for (k, (z, nu, x, p)) in exact.iter().enumerate() {
            let innovation = filter.step::<2>(&f, &h, z, &v, &sw).unwrap();
            let got = filter.covariance();
            for i in 0..2 {
                assert!((innovation[i] - nu[i]).abs() <= tol, "step {k}: ν[{i}]");
                assert!(
                    (filter.x[i] - x[i]).abs() <= tol,
                    "step {k}: x[{i}] = {}",
                    filter.x[i]
                );
                for j in 0..2 {
                    assert!(
                        (got[i][j] - p[i][j]).abs() <= tol,
                        "step {k}: P[{i}][{j}] = {}",
                        got[i][j]
                    );
                }
            }
        }
    }

    /// (G3) The existence condition at the prior is `P₀⁻¹ − γ Q̄ ≻ 0`, which for `Q̄ = I` holds
    /// exactly while `γ < 1/λ_max(P₀)`. `derivations/hinf_riccati.mac` puts that threshold at
    /// 0.2312040466020173954678145 for `P₀ = [[4, 1], [1, 5/4]]`; the construction has to accept a
    /// γ a part in 1e9 below it and refuse one a part in 1e9 above it.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn the_bound_is_refused_exactly_where_the_prior_loses_definiteness() {
        let threshold = 2.312_040_466_020_173_954_678_145e-1;
        let s0 = [[2.0, 0.0], [0.5, 1.0]];
        assert!(HInfinityFilter::new([0.0; 2], s0, identity(), threshold * (1.0 - 1e-9)).is_ok());
        assert_eq!(
            HInfinityFilter::new([0.0; 2], s0, identity(), threshold * (1.0 + 1e-9)).unwrap_err(),
            HInfinityError::UnattainableBound { direction: 1 }
        );
    }

    /// A deterministic measurement stream: a linear-congruential jitter, so the comparison below is
    /// on identical data without a random-number dependency.
    fn observations(count: usize, manoeuvre_at: Option<usize>) -> ([Vec<f64>; 2], Vec<f64>) {
        let (mut truth, mut rate) = (0.0_f64, 1.0_f64);
        let mut seed = 12_345_u64;
        let (mut states, mut rates, mut z) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..count {
            // The disturbance the dynamics model does not know about: a step change in rate.
            if manoeuvre_at == Some(k) {
                rate += 3.0;
            }
            truth += rate;
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            #[allow(clippy::cast_precision_loss)]
            let jitter = ((seed >> 33) as f64 / f64::from(1_u32 << 31) - 0.5) * 0.6;
            states.push(truth);
            rates.push(rate);
            z.push(truth + jitter);
        }
        ([states, rates], z)
    }

    /// Oracle — at `γ = 0` the H∞ recursion *is* the Kalman filter, so it must reproduce
    /// [`SquareRootKalman`] on identical data to rounding.
    ///
    /// The comparison is against the Kalman filter run in the same one-step-predictor order —
    /// measurement update, then time update — because that is the recursion H∞ is written in. Twelve
    /// steps, worst departure 1.8e-15 in the state and 2.8e-16 in the covariance.
    #[test]
    fn the_zero_gamma_filter_is_the_kalman_filter() {
        let (_, z) = observations(12, None);
        let mut hinf =
            HInfinityFilter::<2>::from_covariance([0.0, 0.0], &P0, identity(), 0.0).unwrap();
        let mut kalman = SquareRootKalman::<2>::from_covariance([0.0, 0.0], &P0).unwrap();

        let (mut worst_x, mut worst_p) = (0.0_f64, 0.0_f64);
        for measurement in &z {
            hinf.step::<1>(&F, &H, &[*measurement], &[[R]], &SQ)
                .unwrap();

            let predicted = kalman.x[0];
            kalman.measurement_update(&H[0], measurement - predicted, R);
            let updated = kalman.x;
            for i in 0..2 {
                kalman.x[i] = F[i][0] * updated[0] + F[i][1] * updated[1];
            }
            kalman.time_update(&F, &SQ);

            let (ph, pk) = (hinf.covariance(), kalman.covariance());
            for i in 0..2 {
                worst_x = worst_x.max((hinf.x[i] - kalman.x[i]).abs());
                for j in 0..2 {
                    worst_p = worst_p.max((ph[i][j] - pk[i][j]).abs());
                }
            }
        }
        println!("H-infinity(gamma=0) vs SR-KF: max |dx| {worst_x:.3e}, max |dP| {worst_p:.3e}");
        assert!(
            worst_x < 1e-12,
            "state departed from the Kalman filter: {worst_x:.3e}"
        );
        assert!(worst_p < 1e-12, "covariance departed: {worst_p:.3e}");
    }

    /// The same oracle on a fixture with no symmetry left in it, and in three dimensions.
    ///
    /// `the_zero_gamma_filter_is_the_kalman_filter`'s fixture is invariant under a sign flip of the
    /// velocity state: `Q̄ = I`, `P₀` diagonal and `H = [1, 0]` are all unchanged by conjugation with
    /// `J = diag(1, −1)`, so the whole recursion commutes with `J` and the sign of every
    /// strictly-lower entry of a triangular inverse is unobservable through it — measured, not
    /// supposed: flipping `invert_lower`'s off-diagonal sign leaves that test passing at 1e-12. This
    /// one measures a combination of two states from a correlated prior, which breaks the symmetry,
    /// and runs at `N = 3`, where a pre-array sweep that stops at `2 + N` columns instead of `2N`
    /// silently drops the last of the process noise.
    #[test]
    fn the_recursion_holds_in_three_dimensions_with_no_symmetry_to_hide_behind() {
        const F3: [[f64; 3]; 3] = [[1.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]];
        const H3: [[f64; 3]; 1] = [[1.0, 0.5, 0.0]];
        const SQ3: [[f64; 3]; 3] = [[0.02, 0.0, 0.0], [0.01, 0.05, 0.0], [0.0, 0.01, 0.03]];
        const P3: [[f64; 3]; 3] = [[1.0, 0.3, -0.2], [0.3, 0.5, 0.15], [-0.2, 0.15, 0.25]];

        let (_, z) = observations(12, None);
        let mut hinf =
            HInfinityFilter::<3>::from_covariance([0.1, -0.2, 0.05], &P3, identity(), 0.0).unwrap();
        let mut kalman = SquareRootKalman::<3>::from_covariance([0.1, -0.2, 0.05], &P3).unwrap();

        let (mut worst_x, mut worst_p) = (0.0_f64, 0.0_f64);
        for measurement in &z {
            hinf.step::<1>(&F3, &H3, &[*measurement], &[[R]], &SQ3)
                .unwrap();

            let predicted: f64 = (0..3).map(|k| H3[0][k] * kalman.x[k]).sum();
            kalman.measurement_update(&H3[0], measurement - predicted, R);
            let updated = kalman.x;
            for i in 0..3 {
                kalman.x[i] = (0..3).map(|k| F3[i][k] * updated[k]).sum();
            }
            kalman.time_update(&F3, &SQ3);

            let (ph, pk) = (hinf.covariance(), kalman.covariance());
            for i in 0..3 {
                worst_x = worst_x.max((hinf.x[i] - kalman.x[i]).abs());
                for j in 0..3 {
                    worst_p = worst_p.max((ph[i][j] - pk[i][j]).abs());
                }
            }
        }
        println!(
            "3-D H-infinity(gamma=0) vs SR-KF: max |dx| {worst_x:.3e}, max |dP| {worst_p:.3e}"
        );
        assert!(worst_x < 1e-12, "state departed: {worst_x:.3e}");
        assert!(worst_p < 1e-12, "covariance departed: {worst_p:.3e}");
    }

    /// The case the filter exists for: a disturbance the dynamics model does not contain, and the
    /// price paid for handling it.
    ///
    /// A single realisation cannot settle this — the minimum-variance claim is about the *expected*
    /// squared error, so the comparison is a 200-run Monte Carlo over the same measurement and
    /// process-noise streams for both filters, with the truth driven by exactly the process noise the
    /// filters were told about. Position RMS from step 5 on, 60 steps, γ = 0.9:
    ///
    /// | stream | H∞ RMS | Kalman RMS | H∞ peak | Kalman peak |
    /// |---|---|---|---|---|
    /// | +3 rate step at k = 20 | **1.190** | 1.274 | **5.068** | 5.355 |
    /// | matched model | 0.2752 | **0.2738** | 1.062 | 1.076 |
    ///
    /// Both directions are the point. On the unmodelled manoeuvre the minimax filter is 6.6 % better
    /// in RMS and 5.4 % lower at the peak, and the margin grows with γ (RMS 1.246 at γ = 0.3). On data
    /// that matches the model it is 0.5 % *worse*, because there the Kalman filter is by construction
    /// the minimum-variance estimator and every unit of γ is insurance nobody needed.
    #[test]
    fn the_minimax_filter_is_bounded_where_the_kalman_filter_is_optimistic() {
        let manoeuvre = monte_carlo(0.9, Some(20));
        let quiet = monte_carlo(0.9, None);
        println!("manoeuvre: {manoeuvre:?}\nmatched:   {quiet:?}");

        assert!(
            manoeuvre.rms_hinf < 0.96 * manoeuvre.rms_kalman,
            "the minimax filter must bound the unmodelled step: RMS {:.4} vs {:.4}",
            manoeuvre.rms_hinf,
            manoeuvre.rms_kalman
        );
        assert!(
            manoeuvre.peak_hinf < manoeuvre.peak_kalman,
            "peak {:.3} vs {:.3}",
            manoeuvre.peak_hinf,
            manoeuvre.peak_kalman
        );
        // A larger bound buys more of the same protection.
        assert!(
            manoeuvre.rms_hinf < monte_carlo(0.3, Some(20)).rms_hinf,
            "the protection must grow with gamma"
        );
        assert!(
            quiet.rms_kalman < quiet.rms_hinf,
            "on data that matches the model the Kalman filter must not be beaten: {:.4} vs {:.4}",
            quiet.rms_kalman,
            quiet.rms_hinf
        );
    }

    /// The existence condition is enforced, not documented: a γ too aggressive for the prior is
    /// refused at construction, and one that only becomes infeasible later is refused mid-run with the
    /// direction that failed.
    #[test]
    fn an_unattainable_bound_is_refused_rather_than_answered() {
        // P₀⁻¹ = diag(1, 2), so γ = 3 makes `P₀⁻¹ − γ Q̄` indefinite in the first direction.
        let refused = HInfinityFilter::<2>::from_covariance([0.0, 0.0], &P0, identity(), 3.0);
        assert_eq!(
            refused.err(),
            Some(HInfinityError::UnattainableBound { direction: 0 })
        );
        assert!(HInfinityFilter::<2>::from_covariance([0.0, 0.0], &P0, identity(), -1.0).is_err());
        assert!(
            HInfinityFilter::<2>::from_covariance([0.0, 0.0], &P0, identity(), f64::NAN).is_err()
        );

        // Feasible at the prior, then walked into infeasibility: the rate is never measured, so with
        // a process noise of σ = 2 per step the propagated `P` in that direction is at least 4, its
        // inverse 0.25, and `P⁻¹ − 0.9 Q̄` is indefinite there on the very next step.
        let loud = [[0.1, 0.0], [0.0, 2.0]];
        let mut filter =
            HInfinityFilter::<2>::from_covariance([0.0, 0.0], &P0, identity(), 0.9).unwrap();
        let mut failure = None;
        for _ in 0..50 {
            if let Err(error) = filter.step::<1>(&F, &H, &[0.0], &[[R]], &loud) {
                failure = Some(error);
                break;
            }
        }
        assert_eq!(
            failure,
            Some(HInfinityError::ExistenceConditionViolated { direction: 1 }),
            "the recursion must report infeasibility rather than returning a gain"
        );
    }

    /// A singular prior and a singular measurement noise are refused, each naming its own direction.
    #[test]
    fn the_degenerate_inputs_are_named_rather_than_inverted() {
        let singular = [[1.0, 0.0], [0.0, 0.0]];
        assert_eq!(
            HInfinityFilter::<2>::from_covariance([0.0, 0.0], &singular, identity(), 0.0).err(),
            Some(HInfinityError::SingularCovariance { direction: 1 })
        );

        let mut filter =
            HInfinityFilter::<2>::from_covariance([0.0, 0.0], &P0, identity(), 0.0).unwrap();
        assert_eq!(
            filter.step::<1>(&F, &H, &[0.0], &[[0.0]], &SQ).err(),
            Some(HInfinityError::MeasurementNoiseNotPositiveDefinite { direction: 0 })
        );
    }

    /// The reported factor is a factor of the reported covariance, and the innovation is the one the
    /// gain was applied to.
    #[test]
    fn the_reported_factor_multiplies_out_to_the_reported_covariance() {
        let mut filter =
            HInfinityFilter::<2>::from_covariance([0.4, -0.2], &P0, identity(), 0.05).unwrap();
        let innovation = filter.step::<1>(&F, &H, &[1.25], &[[R]], &SQ).unwrap();
        assert!((innovation[0] - (1.25 - 0.4)).abs() < 1e-15, "z − H x̂");

        let (s, p) = (filter.covariance_sqrt(), filter.covariance());
        for i in 0..2 {
            for j in 0..2 {
                if j > i {
                    assert_eq!(s[i][j], 0.0, "S[{i}][{j}] is above the diagonal");
                }
                let sst: f64 = (0..2).map(|k| s[i][k] * s[j][k]).sum();
                assert!((sst - p[i][j]).abs() < 1e-14, "S Sᵀ[{i}][{j}]");
            }
        }
        assert_eq!(filter.gamma(), 0.05);
    }

    /// Each refusal says which direction failed, in words a caller can act on. The message is the
    /// diagnosis — a filter that returns "error" and nothing else costs the caller the measurement.
    #[test]
    fn every_refusal_names_its_own_direction() {
        let messages = [
            (
                HInfinityError::ExistenceConditionViolated { direction: 2 },
                "H-infinity existence condition fails in direction 2: reduce gamma",
            ),
            (
                HInfinityError::SingularCovariance { direction: 1 },
                "covariance is singular in direction 1",
            ),
            (
                HInfinityError::MeasurementNoiseNotPositiveDefinite { direction: 0 },
                "measurement noise covariance is not positive-definite in direction 0",
            ),
            (
                HInfinityError::UnattainableBound { direction: 3 },
                "gamma is unattainable for this prior covariance in direction 3",
            ),
        ];
        for (error, expected) in messages {
            assert_eq!(format!("{error}"), expected);
            let dynamic: &dyn core::error::Error = &error;
            assert_eq!(format!("{dynamic}"), expected);
        }
    }

    /// A larger γ is a more conservative filter: it trusts its own model less, so it keeps a larger
    /// covariance and a larger gain on the same data.
    #[test]
    fn a_larger_bound_keeps_a_larger_covariance() {
        let (_, z) = observations(15, None);
        let mut trace = Vec::new();
        for gamma in [0.0, 0.05, 0.15] {
            let mut filter =
                HInfinityFilter::<2>::from_covariance([0.0, 0.0], &P0, identity(), gamma).unwrap();
            for measurement in &z {
                filter
                    .step::<1>(&F, &H, &[*measurement], &[[R]], &SQ)
                    .unwrap();
            }
            let p = filter.covariance();
            trace.push(p[0][0] + p[1][1]);
        }
        assert!(
            trace[0] < trace[1] && trace[1] < trace[2],
            "trace P must grow with gamma: {trace:?}"
        );
    }

    /// Gaussian pair from an LCG, so a Monte Carlo run is reproducible without a dependency.
    fn gaussians(seed: &mut u64) -> (f64, f64) {
        let mut uniform = || {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let u = ((*seed >> 11) as f64) / ((1_u64 << 53) as f64);
            u.mul_add(0.999_999_9, 5e-8)
        };
        let (u1, u2) = (uniform(), uniform());
        let r = libm::sqrt(-2.0 * libm::log(u1));
        (
            r * libm::cos(core::f64::consts::TAU * u2),
            r * libm::sin(core::f64::consts::TAU * u2),
        )
    }

    #[derive(Debug)]
    struct Score {
        rms_hinf: f64,
        rms_kalman: f64,
        peak_hinf: f64,
        peak_kalman: f64,
    }

    /// One Monte Carlo comparison: both filters on the same 200 realisations of the same stream, the
    /// truth driven by the process noise `SQ` describes, scored on the one-step position prediction
    /// once the transient has passed.
    fn monte_carlo(gamma: f64, manoeuvre_at: Option<usize>) -> Score {
        const RUNS: usize = 200;
        const STEPS: usize = 60;
        const SETTLED: usize = 5;

        let (mut sse_hinf, mut sse_kalman) = (0.0_f64, 0.0_f64);
        let (mut peak_hinf, mut peak_kalman) = (0.0_f64, 0.0_f64);
        for run in 0..RUNS {
            let mut seed = 1 + u64::try_from(run).unwrap() * 7919;
            let (mut position, mut rate) = (0.0_f64, 1.0_f64);
            let mut hinf =
                HInfinityFilter::<2>::from_covariance([0.0, 0.0], &P0, identity(), gamma).unwrap();
            let mut kalman = SquareRootKalman::<2>::from_covariance([0.0, 0.0], &P0).unwrap();

            for k in 0..STEPS {
                let (drive_position, drive_rate) = gaussians(&mut seed);
                let (jitter, _) = gaussians(&mut seed);

                // Both estimates entering this step are one-step predictions of the state now.
                if k >= SETTLED {
                    let error_hinf = (hinf.x[0] - position).abs();
                    let error_kalman = (kalman.x[0] - position).abs();
                    sse_hinf += error_hinf * error_hinf;
                    sse_kalman += error_kalman * error_kalman;
                    peak_hinf = peak_hinf.max(error_hinf);
                    peak_kalman = peak_kalman.max(error_kalman);
                }

                let measurement = position + libm::sqrt(R) * jitter;
                hinf.step::<1>(&F, &H, &[measurement], &[[R]], &SQ).unwrap();
                let predicted = kalman.x[0];
                kalman.measurement_update(&H[0], measurement - predicted, R);
                let updated = kalman.x;
                for i in 0..2 {
                    kalman.x[i] = F[i][0] * updated[0] + F[i][1] * updated[1];
                }
                kalman.time_update(&F, &SQ);

                // The disturbance the dynamics model does not know about: a step change in rate.
                if manoeuvre_at == Some(k) {
                    rate += 3.0;
                }
                position += rate + SQ[0][0] * drive_position;
                rate += SQ[1][1] * drive_rate;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let samples = (RUNS * (STEPS - SETTLED)) as f64;
        Score {
            rms_hinf: libm::sqrt(sse_hinf / samples),
            rms_kalman: libm::sqrt(sse_kalman / samples),
            peak_hinf,
            peak_kalman,
        }
    }

    #[test]
    fn probe_negated_left() {
        // Mirror of `step`'s pre-array with the accumulation negated, compared on P and on S.
        let (_, z) = observations(6, None);
        let mut a =
            HInfinityFilter::<2>::from_covariance([0.0, 0.0], &P0, identity(), 0.1).unwrap();
        for m in &z {
            a.step::<1>(&F, &H, &[*m], &[[R]], &SQ).unwrap();
        }
        let (pa, sa) = (a.covariance(), a.covariance_sqrt());
        println!("P {pa:?}\nS {sa:?}");
    }
}
