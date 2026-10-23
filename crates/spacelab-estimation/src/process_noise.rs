//! Discrete process noise: turning the continuous spectral density physics gives you into the `Q`
//! every filter in this crate demands.
//!
//! # The gap this closes
//!
//! [`SquareRootKalman::time_update`](crate::kalman::SquareRootKalman::time_update) and
//! [`SquareRootUnscentedKalman::predict`](crate::srukf::SquareRootUnscentedKalman::predict) both take
//! a `process_noise_sqrt` from the caller. What the caller actually knows is a *continuous* quantity —
//! an unmodelled acceleration of so many m·s⁻² per √Hz, i.e. a spectral density in m²·s⁻³ — and the
//! filter wants the covariance that density accumulates over one step. Without something to bridge
//! the two, `Q` gets tuned by hand until the residuals look acceptable, which is the most common way
//! a filter ends up with a covariance that is confidently wrong. [`crate::consistency`] exists to
//! detect exactly that failure; this module is the half that stops causing it.
//!
//! # Van Loan rather than a series
//!
//! For linear time-invariant dynamics `ẋ = F x + w`, `E[w(t)w(s)ᵀ] = Q_c δ(t−s)`, the discrete
//! covariance over a step is
//!
//! ```text
//! Q_d = ∫₀^Δt e^{F s} Q_c e^{Fᵀ s} ds
//! ```
//!
//! [`van_loan`] evaluates that integral without doing it, by exponentiating one `2n × 2n` block
//! matrix ([van Loan 1978]). It is exact for LTI dynamics — not a truncated series whose error is a
//! function of `‖F‖Δt` that nobody checks. `derivations/process_noise.mac` proves the construction
//! against the integral above symbolically, on a case with no exploitable structure: both the `Φ` and
//! the `Q_d` residual are identically zero.
//!
//! # Why the closed forms are only in the tests
//!
//! The closed form for a DMC step is available and this module does not use it. That is a
//! conditioning decision, and it was measured rather than assumed. The DMC position variance is a sum
//! of four terms of alternating sign; at `τ = 200 s`, `q = 1e-12`, `Δt = 60 s` they are
//!
//! ```text
//!  0.0721901…  −0.0462371…  −0.0288  +0.00288  =  3.303985…e-5
//! ```
//!
//! and the cancellation worsens as `(Δt/τ)⁻⁴`. Evaluating that closed form in `f64` against a
//! 50-digit reference (`mpmath`) gives:
//!
//! ```text
//! Δt/τ     largest term / result     relative error of the f64 closed form
//! 1.0                       33.4                                  1.6e-16
//! 0.3                       2185                                  1.9e-13
//! 0.1                     1.9e+5                                  1.8e-10
//! 0.03                    2.4e+7                                  4.3e-09
//! 0.01                    2.0e+9                                  6.3e-06
//! 0.003                   2.5e+11                                 4.9e-04
//! ```
//!
//! At `Δt/τ = 0.01` — a one-minute step against a two-hour correlation time, an ordinary filter
//! setting — the closed form has lost six significant digits. [`van_loan`] reaches the same numbers
//! through a scaled-and-squared matrix exponential, which contains no such subtraction, and holds
//! `1e-16`-ish there. So the closed forms serve as *oracles* in the tests, evaluated at parameters
//! that keep them accurate, and the shipped path is the better-conditioned one.
//!
//! The same measurement is why the DMC test's reference values come from `mpmath` and not from
//! Maxima's own `float()`: at `Δt/τ = 0.3` Maxima's numeric evaluation of its own symbolic result is
//! itself only good to `2e-13`, for exactly this reason.
//!
//! # What is here
//!
//! * [`matrix_exp`] — scaling and squaring with a Padé approximant; a general primitive the workspace
//!   did not have.
//! * [`van_loan`] — `(Φ, Q_d)` for an LTI step.
//! * [`state_noise_compensation`] — SNC, piecewise-constant white acceleration on a
//!   position/velocity state, the classical orbit-determination form.
//! * [`dynamic_model_compensation`] — DMC, a first-order Gauss–Markov acceleration with a correlation
//!   time, state-augmented. This is the form that works when the error being absorbed is a
//!   mis-modelled drag or SRP coefficient, because that error is *correlated*, not white; modelling it
//!   as white lets the covariance shrink around a wrong answer.
//! * [`rotate_position_velocity`] — move an SNC block from the RTN frame the sigmas are natural in to
//!   the frame the filter runs in.
//!
//! [van Loan 1978]: C. F. van Loan, "Computing integrals involving the matrix exponential",
//! IEEE Trans. Automat. Control 23(3), 1978.

// Matrix code indexes several arrays by one i/j/k, and the filter symbols (F, Q, S, Φ) are one letter
// by universal convention.
#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]

use crate::unscented::cholesky;

/// Why a process-noise construction was rejected.
///
/// Each variant names a specific violated precondition rather than collapsing to "bad input", because
/// the three are confused in practice: a negative step is a sign error at the call site, a
/// non-positive correlation time is a DMC parameter that was never set, and a non-finite input is
/// usually a NaN that arrived from somewhere else entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessNoiseError {
    /// An entry of `F`, `Q_c`, or a parameter was NaN or infinite.
    NonFinite,
    /// The step was negative. `Q_d` is the noise *accumulated* over a step; running a filter backwards
    /// does not un-accumulate it, so a backward pass uses `|Δt|` and the sign is the caller's error.
    NegativeStep,
    /// A Gauss–Markov correlation time was zero or negative. The limit `τ → 0` is white noise, which
    /// is [`state_noise_compensation`], not a degenerate DMC.
    NonPositiveCorrelationTime,
    /// A standard deviation was negative.
    NegativeSigma,
    /// The Padé denominator was singular, so the matrix exponential could not be formed. Unreachable
    /// for a finite input after scaling; reported rather than silently returning a wrong matrix.
    ExponentialFailed,
}

/// A discretised step: the state transition matrix `Φ` and the discrete process noise `Q_d`, in that
/// order, both `N × N`.
pub type DiscreteStep<const N: usize> = ([[f64; N]; N], [[f64; N]; N]);

/// Padé order. Six is the classical choice: with the argument scaled to `‖A‖∞ ≤ 1/2` the `[6/6]`
/// approximant is accurate to well below `f64` rounding, and each extra order costs a matrix product
/// that buys nothing.
const PADE_ORDER: u8 = 6;

/// The scaled argument's target infinity-norm. The Padé error bound is a function of this, so it is
/// the one number that sets the accuracy of [`matrix_exp`].
const PADE_THETA: f64 = 0.5;

/// Guard on the number of squarings. `‖A‖∞` above `2^1024` is not representable, so this bound is
/// only reachable through an input that is already meaningless.
const MAX_SQUARINGS: u32 = 1024;

/// The matrix exponential `e^A`, by scaling and squaring with a Padé approximant.
///
/// `A` is scaled by `2^{-s}` until its infinity-norm is at most `PADE_THETA`, approximated by the
/// `[6/6]` Padé rational — the one regime where that rational is accurate — and the result squared
/// back `s` times.
///
/// Returns `None` for a non-finite input or a singular Padé denominator.
///
/// # Accuracy, as measured
///
/// Scaling and squaring is textbook-suspect on two counts — a defective (non-diagonalisable) matrix,
/// and widely separated eigenvalues whose repeated squaring amplifies the rounding of the small one.
/// Both are tested here against closed forms, and both come back at `1e-13` or better.
///
/// The limitation that *does* bite is different, and it is entrywise: the exponential is accurate in
/// norm, so the relative accuracy of a small entry is set by the largest entry of the same matrix.
/// In a DMC block spanning six orders of magnitude between `Q_rr` and `Q_ww`, the acceleration
/// variance carries a measured `3e-13` relative error at `Δt/τ = 10` while the position variance
/// carries `1e-16`. That is a property of the method, not a bug, and it is pinned by
/// `van_loan_accuracy_degrades_only_with_the_dynamic_range_of_the_block`.
#[must_use]
pub fn matrix_exp<const N: usize>(a: &[[f64; N]; N]) -> Option<[[f64; N]; N]> {
    let mut norm = 0.0f64;
    for i in 0..N {
        let mut row = 0.0;
        for j in 0..N {
            if !a[i][j].is_finite() {
                return None;
            }
            row += libm::fabs(a[i][j]);
        }
        if row > norm {
            norm = row;
        }
    }

    // Smallest `s` with `norm / 2^s <= theta`, found by halving rather than through `log2` and a
    // float-to-integer cast: halving an `f64` is exact, so this cannot land one squaring off.
    let mut squarings: u32 = 0;
    let mut reduced = norm;
    while reduced > PADE_THETA {
        reduced *= 0.5;
        squarings += 1;
        if squarings > MAX_SQUARINGS {
            return None;
        }
    }
    let scale = libm::exp2(-f64::from(squarings));

    let mut scaled = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            scaled[i][j] = a[i][j] * scale;
        }
    }

    // Numerator and denominator of the [q/q] Padé approximant share the powers of A and differ only
    // in the sign of the odd ones, so both are accumulated in one pass.
    let mut numerator = identity::<N>();
    let mut denominator = identity::<N>();
    let mut power = identity::<N>();
    let mut c = 1.0;
    let q = f64::from(PADE_ORDER);
    for k in 1..=PADE_ORDER {
        let kf = f64::from(k);
        c *= (q - kf + 1.0) / ((2.0 * q - kf + 1.0) * kf);
        power = matmul(&power, &scaled);
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        for i in 0..N {
            for j in 0..N {
                numerator[i][j] += c * power[i][j];
                denominator[i][j] += sign * c * power[i][j];
            }
        }
    }

    let mut result = solve(&denominator, &numerator)?;
    for _ in 0..squarings {
        result = matmul(&result, &result);
    }
    Some(result)
}

/// The state transition matrix and discrete process noise for one step of linear time-invariant
/// dynamics: `ẋ = F x + w` with `E[w(t)w(s)ᵀ] = Q_c δ(t−s)`.
///
/// Returns `(Φ, Q_d)` with `Φ = e^{F Δt}` and `Q_d = ∫₀^Δt e^{F s} Q_c e^{Fᵀ s} ds`, both exact for
/// LTI dynamics.
///
/// `M` must be `2 * N` — the block matrix Van Loan exponentiates is twice the state dimension, and
/// stable Rust cannot compute that from `N`, so the caller states it and the constructor checks it.
/// This mirrors the `L = 2N + 1` sigma-point parameter of
/// [`SquareRootUnscentedKalman`](crate::srukf::SquareRootUnscentedKalman).
///
/// # Errors
///
/// [`ProcessNoiseError::NegativeStep`] for `dt < 0`, [`ProcessNoiseError::NonFinite`] for a non-finite
/// entry or step, [`ProcessNoiseError::ExponentialFailed`] if the exponential could not be formed.
///
/// # Panics
///
/// If `M != 2 * N`. That is a mismatch between two const parameters at the call site, i.e. a
/// programming error rather than bad data, so it fails loudly at once instead of being threaded
/// through the error type.
pub fn van_loan<const N: usize, const M: usize>(
    f: &[[f64; N]; N],
    q_continuous: &[[f64; N]; N],
    dt: f64,
) -> Result<DiscreteStep<N>, ProcessNoiseError> {
    assert!(M == 2 * N, "van_loan: M must equal 2 * N");
    if !dt.is_finite() {
        return Err(ProcessNoiseError::NonFinite);
    }
    if dt < 0.0 {
        return Err(ProcessNoiseError::NegativeStep);
    }
    for i in 0..N {
        for j in 0..N {
            if !f[i][j].is_finite() || !q_continuous[i][j].is_finite() {
                return Err(ProcessNoiseError::NonFinite);
            }
        }
    }

    // C = [[-F, Q_c], [0, Fᵀ]] · Δt.
    let mut c = [[0.0; M]; M];
    for i in 0..N {
        for j in 0..N {
            c[i][j] = -f[i][j] * dt;
            c[i][j + N] = q_continuous[i][j] * dt;
            c[i + N][j + N] = f[j][i] * dt;
        }
    }

    let x = matrix_exp::<M>(&c).ok_or(ProcessNoiseError::ExponentialFailed)?;

    // Φ is the transpose of the lower-right block; Q_d is Φ times the upper-right block.
    let mut phi = [[0.0; N]; N];
    let mut upper_right = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            phi[i][j] = x[j + N][i + N];
            upper_right[i][j] = x[i][j + N];
        }
    }
    let mut q_discrete = matmul(&phi, &upper_right);

    // Q_d is symmetric in exact arithmetic. Averaging removes the rounding asymmetry at the point of
    // *construction*, which is a different act from symmetrising a covariance a filter has been
    // updating — there is no divergence being hidden, and the residual asymmetry is measured in the
    // tests so a real failure would still show.
    for i in 0..N {
        for j in 0..i {
            let mean = f64::midpoint(q_discrete[i][j], q_discrete[j][i]);
            q_discrete[i][j] = mean;
            q_discrete[j][i] = mean;
        }
    }

    Ok((phi, q_discrete))
}

/// State noise compensation: the discrete covariance of a piecewise-constant white acceleration
/// acting on a position/velocity state over a step.
///
/// `sigma_acceleration` is the *square root of the spectral density* per axis, in m·s⁻³ᐟ², expressed
/// in whatever frame the caller means — apply [`rotate_position_velocity`] to move an RTN-specified
/// noise into an inertial one. The state order is `(r, v)` with all three position components first.
///
/// The per-axis block is the classical `q · [[Δt³/3, Δt²/2], [Δt²/2, Δt]]`, which
/// `derivations/process_noise.mac` derives from the defining integral rather than quoting.
///
/// # Errors
///
/// [`ProcessNoiseError::NegativeStep`], [`ProcessNoiseError::NegativeSigma`] or
/// [`ProcessNoiseError::NonFinite`].
pub fn state_noise_compensation(
    dt: f64,
    sigma_acceleration: [f64; 3],
) -> Result<[[f64; 6]; 6], ProcessNoiseError> {
    if !dt.is_finite() || !sigma_acceleration.iter().all(|s| s.is_finite()) {
        return Err(ProcessNoiseError::NonFinite);
    }
    if dt < 0.0 {
        return Err(ProcessNoiseError::NegativeStep);
    }
    if sigma_acceleration.iter().any(|&s| s < 0.0) {
        return Err(ProcessNoiseError::NegativeSigma);
    }

    let (rr, rv, vv) = (dt * dt * dt / 3.0, 0.5 * dt * dt, dt);
    let mut q = [[0.0; 6]; 6];
    for axis in 0..3 {
        let density = sigma_acceleration[axis] * sigma_acceleration[axis];
        q[axis][axis] = density * rr;
        q[axis][axis + 3] = density * rv;
        q[axis + 3][axis] = density * rv;
        q[axis + 3][axis + 3] = density * vv;
    }
    Ok(q)
}

/// Rotate a position/velocity process-noise block by `rotation`, applied to both halves:
/// `Q' = A Q Aᵀ` with `A = diag(R, R)`.
///
/// The use is [`state_noise_compensation`], whose sigmas are natural in the RTN frame (radial,
/// along-track and cross-track unmodelled accelerations are physically different sizes) while the
/// filter runs in an inertial one. `rotation` maps a vector *from* the frame the sigmas are in *to*
/// the frame the filter uses.
#[must_use]
pub fn rotate_position_velocity(q: &[[f64; 6]; 6], rotation: &[[f64; 3]; 3]) -> [[f64; 6]; 6] {
    let mut a = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] = rotation[i][j];
            a[i + 3][j + 3] = rotation[i][j];
        }
    }
    let aq = matmul(&a, q);
    let mut out = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut acc = 0.0;
            for k in 0..6 {
                acc += aq[i][k] * a[j][k];
            }
            out[i][j] = acc;
        }
    }
    out
}

/// Dynamic model compensation: `(Φ, Q_d)` for a nine-state `(r, v, w)` where the acceleration `w` is a
/// first-order Gauss–Markov process, `ẇ = −w/τ + u`, per axis.
///
/// `correlation_time` is `τ` per axis in seconds and `sigma_acceleration` is the *stationary* standard
/// deviation of `w` in m·s⁻². The forcing density that produces that stationary variance is
/// `q = 2σ²/τ` — a relation `derivations/process_noise.mac` checks as the `Δt → ∞` limit of the
/// process's own variance rather than asserting.
///
/// # Why this rather than SNC
///
/// A mis-modelled drag or solar-radiation-pressure coefficient does not produce white acceleration
/// error; it produces an error that persists for as long as the mis-modelling does. Represented as
/// white noise ([`state_noise_compensation`]) the filter treats each step's error as independent, so
/// the covariance contracts while the mean stays biased. Given a correlation time, the augmented state
/// absorbs the error instead, and the covariance stays honest. `τ` is the physical statement of how
/// long the error is believed to persist and is the parameter worth tuning.
///
/// Computed through [`van_loan`], so this shares one implementation with the general case rather than
/// carrying a second, transcribed copy of the algebra.
///
/// # What the returned `Φ` is, and is not
///
/// `F` here is purely kinematic — `ṙ = v`, `v̇ = w` — with **no gravity gradient**, so the returned
/// `Φ` is the transition of the augmented model alone. In an orbit determination filter the
/// position/velocity block must come from the variational equations
/// (`spacelab_propagation::variational`), and only `Q_d` is taken from here. That is the standard
/// arrangement and it is an approximation with a name: over one step the gravity gradient's
/// contribution to the *noise* integral is negligible even where its contribution to the *transition*
/// is not.
///
/// # Verified end to end
///
/// `tests/process_noise_consistency.rs` runs both this and [`state_noise_compensation`] against a
/// Gauss–Markov coefficient error over a nine-hour arc with realistic tracking gaps. Measured, at 200
/// Monte-Carlo runs: DMC lands at NEES 6.31 inside the six-degree-of-freedom band `[5.29, 6.76]` at
/// its declared `τ` and `σ`, with 13.52 m RMS position error. A swept white-acceleration `Q` reaches
/// the band at exactly one point of a seventeen-point sweep across four decades — 161 700 at
/// `σ_a = 1e-7`, 16.7 a quarter-decade below the crossing and 1.96 a quarter-decade above — and that
/// one tuning still costs 11% more position error.
///
/// # Errors
///
/// [`ProcessNoiseError::NonPositiveCorrelationTime`] for `τ ≤ 0` (the `τ → 0` limit is white noise —
/// use [`state_noise_compensation`]), plus the step and finiteness conditions of [`van_loan`].
pub fn dynamic_model_compensation(
    dt: f64,
    correlation_time: [f64; 3],
    sigma_acceleration: [f64; 3],
) -> Result<DiscreteStep<9>, ProcessNoiseError> {
    if !correlation_time.iter().all(|t| t.is_finite())
        || !sigma_acceleration.iter().all(|s| s.is_finite())
    {
        return Err(ProcessNoiseError::NonFinite);
    }
    if correlation_time.iter().any(|&t| t <= 0.0) {
        return Err(ProcessNoiseError::NonPositiveCorrelationTime);
    }
    if sigma_acceleration.iter().any(|&s| s < 0.0) {
        return Err(ProcessNoiseError::NegativeSigma);
    }

    let mut f = [[0.0; 9]; 9];
    let mut q_continuous = [[0.0; 9]; 9];
    for axis in 0..3 {
        f[axis][axis + 3] = 1.0; // ṙ = v
        f[axis + 3][axis + 6] = 1.0; // v̇ = w
        f[axis + 6][axis + 6] = -1.0 / correlation_time[axis]; // ẇ = −w/τ
        let sigma = sigma_acceleration[axis];
        q_continuous[axis + 6][axis + 6] = 2.0 * sigma * sigma / correlation_time[axis];
    }
    van_loan::<9, 18>(&f, &q_continuous, dt)
}

/// A square root `S_Q` with `Q = S_Q S_Qᵀ`, ready to hand to
/// [`SquareRootKalman::time_update`](crate::kalman::SquareRootKalman::time_update).
///
/// Returns `None` if `q` is not positive-definite. For the constructions here that means a state
/// direction that receives no noise at all — legitimate, but it needs a rank-deficient factor rather
/// than a Cholesky one, and returning `None` says so instead of producing a factor that is quietly
/// wrong.
#[must_use]
pub fn process_noise_sqrt<const N: usize>(q: &[[f64; N]; N]) -> Option<[[f64; N]; N]> {
    cholesky(q)
}

fn identity<const N: usize>() -> [[f64; N]; N] {
    let mut m = [[0.0; N]; N];
    for i in 0..N {
        m[i][i] = 1.0;
    }
    m
}

fn matmul<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut c = [[0.0; N]; N];
    for i in 0..N {
        for k in 0..N {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..N {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

/// Solve `A X = B` for `X` by Gaussian elimination with partial pivoting. `None` if `A` is singular.
fn solve<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> Option<[[f64; N]; N]> {
    let mut lu = *a;
    let mut x = *b;
    for col in 0..N {
        let mut pivot = col;
        let mut best = libm::fabs(lu[col][col]);
        for row in (col + 1)..N {
            let candidate = libm::fabs(lu[row][col]);
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        if best == 0.0 {
            return None;
        }
        if pivot != col {
            lu.swap(pivot, col);
            x.swap(pivot, col);
        }
        for row in (col + 1)..N {
            let factor = lu[row][col] / lu[col][col];
            if factor == 0.0 {
                continue;
            }
            for k in col..N {
                lu[row][k] -= factor * lu[col][k];
            }
            for k in 0..N {
                x[row][k] -= factor * x[col][k];
            }
        }
    }
    for col in (0..N).rev() {
        for k in 0..N {
            let mut acc = x[col][k];
            for j in (col + 1)..N {
                acc -= lu[col][j] * x[j][k];
            }
            x[col][k] = acc / lu[col][col];
        }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::float_cmp)]

    use super::*;
    fn max_abs_diff<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> f64 {
        let mut worst = 0.0f64;
        for i in 0..N {
            for j in 0..N {
                let d = libm::fabs(a[i][j] - b[i][j]);
                if d > worst {
                    worst = d;
                }
            }
        }
        worst
    }

    /// The exponential of a diagonal matrix is the diagonal of the exponentials — the one case with
    /// an answer that needs no derivation at all.
    /// Every one of these functions validates finiteness with a single `||` over its arguments, and
    /// a suite that only ever passes one bad argument at a time cannot tell that from an `&&`.
    ///
    /// The infinite correlation time is the case that matters and the one that is easy to miss. A
    /// `NaN` there is caught downstream anyway — [`van_loan`] refuses it — so a mutated guard still
    /// produces the right error for the wrong reason. An infinite `τ` is different: `1/τ` is `0`,
    /// every matrix built from it is finite, and the result comes back `Ok`. Silently, that is a
    /// random walk with no decay, which is [`state_noise_compensation`]'s model and not this one.
    #[test]
    fn a_non_finite_input_is_rejected_whichever_argument_carries_it() {
        let sigma = [1.0, 2.0, 3.0];
        let tau = [100.0; 3];
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                state_noise_compensation(bad, sigma).unwrap_err(),
                ProcessNoiseError::NonFinite,
                "SNC accepted a step of {bad} beside finite sigmas"
            );
            assert_eq!(
                state_noise_compensation(10.0, [bad, 2.0, 3.0]).unwrap_err(),
                ProcessNoiseError::NonFinite,
                "SNC accepted a sigma of {bad} beside a finite step"
            );
            assert_eq!(
                dynamic_model_compensation(10.0, [bad, 100.0, 100.0], sigma).unwrap_err(),
                ProcessNoiseError::NonFinite,
                "DMC accepted a correlation time of {bad} beside finite sigmas"
            );
            assert_eq!(
                dynamic_model_compensation(10.0, tau, [bad, 2.0, 3.0]).unwrap_err(),
                ProcessNoiseError::NonFinite,
                "DMC accepted a sigma of {bad} beside a finite correlation time"
            );
        }
    }

    /// Zero process noise is a request, not a malformed input: it is what a caller asks for to run a
    /// filter on pure dynamics, and it sits exactly on the `σ < 0` boundary that rejects the
    /// negative ones.
    #[test]
    fn zero_process_noise_is_a_request_and_not_an_error() {
        let snc = state_noise_compensation(10.0, [0.0; 3]).unwrap();
        assert_eq!(max_abs_diff(&snc, &[[0.0; 6]; 6]), 0.0);
        let (transition, noise) = dynamic_model_compensation(10.0, [100.0; 3], [0.0; 3]).unwrap();
        assert_eq!(max_abs_diff(&noise, &[[0.0; 9]; 9]), 0.0);
        // The transition is the model's, noise or no noise: the acceleration still decays.
        assert!(transition[6][6] < 1.0, "the DMC decay went missing");
    }

    /// `solve` claims partial pivoting, and nothing that reaches it through [`matrix_exp`] ever needs
    /// a swap — the Padé denominator is diagonally dominant by construction. So the pivot search is
    /// exercised only by calling it directly.
    ///
    /// The tie is the second half. Two candidates of equal magnitude have to resolve the same way
    /// every time or the factorisation stops being reproducible, and the convention is the lowest
    /// row index. It is observable only in the last bit — taking the later row instead turns
    /// `0.7` into `0.7000000000000001` — which is precisely why it needs pinning rather than a
    /// tolerance.
    #[test]
    fn solve_pivots_and_breaks_ties_toward_the_first_row() {
        let eye = [[1.0, 0.0], [0.0, 1.0]];

        // Without a swap the pivot is the zero in the corner and the matrix reads as singular.
        let swapped = solve(&[[0.0, 1.0], [1.0, 0.0]], &eye).expect("a permutation is invertible");
        assert_eq!(swapped, [[0.0, 1.0], [1.0, 0.0]]);

        // |1| and |−1| tie in the first column; the exact inverse is [[0.7, −0.3], [0.1, 0.1]].
        let tied = solve(&[[1.0, 3.0], [-1.0, 7.0]], &eye).expect("determinant 10");
        assert_eq!(tied[0][0].to_bits(), 0.7_f64.to_bits());
        assert_eq!(
            tied[0][1].to_bits(),
            (-0.300_000_000_000_000_04_f64).to_bits()
        );
        assert_eq!(tied[1][0].to_bits(), 0.1_f64.to_bits());
        assert_eq!(tied[1][1].to_bits(), 0.1_f64.to_bits());
    }

    /// Both ends of the squaring ladder, which are reachable and were not reached.
    ///
    /// At the bottom, `‖A‖∞ = PADE_THETA` exactly: the comment above the loop promises the *smallest*
    /// `s` with `norm / 2^s ≤ θ`, so the boundary belongs to `s = 0`. Widening the comparison spends
    /// one squaring there, and the only thing that moves is the last bit — 1.648721270700128 becomes
    /// 1.6487212707001284. Neither is `libm::exp(0.5)`, which sits one ulp away from both, so libm
    /// cannot referee this; the branch is what is being pinned, and the ulp bound below is what says
    /// the pinned value is a sane one.
    ///
    /// At the top, exactly `MAX_SQUARINGS` halvings. A nilpotent matrix reaches it with an answer
    /// that is not only finite but exact: `exp(N) = I + N` when `N² = 0`, and both the Padé
    /// approximant and every squaring reproduce that identity in floating point. Off-by-one in the
    /// guard refuses a matrix the method handles perfectly.
    #[test]
    fn the_squaring_ladder_runs_to_both_of_its_documented_boundaries() {
        let at_theta = matrix_exp(&[[PADE_THETA, 0.0], [0.0, -PADE_THETA]]).expect("finite");
        assert_eq!(
            at_theta[0][0].to_bits(),
            1.648_721_270_700_128_f64.to_bits()
        );
        assert_eq!(
            at_theta[1][1].to_bits(),
            6.065_306_597_126_335e-1_f64.to_bits()
        );
        for (got, want) in [
            (at_theta[0][0], libm::exp(PADE_THETA)),
            (at_theta[1][1], libm::exp(-PADE_THETA)),
        ] {
            let ulp = libm::fabs(want) * f64::EPSILON;
            assert!(libm::fabs(got - want) <= ulp, "{got:e} against {want:e}");
        }

        // ‖N‖∞ = 2^1023 needs exactly 1024 halvings to reach θ = 0.5.
        let n = libm::exp2(1023.0);
        let nilpotent = matrix_exp(&[[0.0, n], [0.0, 0.0]]).expect("exp(N) = I + N is finite");
        assert_eq!(nilpotent, [[1.0, n], [0.0, 1.0]]);
    }

    #[test]
    fn the_exponential_of_a_diagonal_is_diagonal() {
        let a = [[0.3, 0.0, 0.0], [0.0, -1.7, 0.0], [0.0, 0.0, 5.0]];
        let e = matrix_exp(&a).unwrap();
        let want = [
            [libm::exp(0.3), 0.0, 0.0],
            [0.0, libm::exp(-1.7), 0.0],
            [0.0, 0.0, libm::exp(5.0)],
        ];
        assert!(max_abs_diff(&e, &want) < 1e-12 * libm::exp(5.0));
    }

    /// A **defective** matrix — a Jordan block, not diagonalisable — whose exponential carries a term
    /// linear in the parameter. Any method that secretly assumes diagonalisability gets this wrong,
    /// and a random well-conditioned test matrix would never reveal it.
    #[test]
    fn the_exponential_of_a_jordan_block_is_the_closed_form() {
        let a = 2.5;
        let m = [[a, 1.0], [0.0, a]];
        let e = matrix_exp(&m).unwrap();
        let ea = libm::exp(a);
        let want = [[ea, ea], [0.0, ea]];
        assert!(max_abs_diff(&e, &want) < 1e-12 * ea);
    }

    /// Widely separated eigenvalues: the regime where scaling and squaring is known to be weakest,
    /// because the repeated squaring amplifies the rounding carried by the small eigenvalue. Built by
    /// similarity so the exact answer is available: `exp(V D V⁻¹) = V exp(D) V⁻¹`.
    #[test]
    fn the_exponential_survives_widely_separated_eigenvalues() {
        let (l1, l2) = (-30.0, 1e-4);
        let v = [[1.0, 1.0], [0.0, 1.0]];
        let v_inv = [[1.0, -1.0], [0.0, 1.0]];
        let d = [[l1, 0.0], [0.0, l2]];
        let m = matmul(&matmul(&v, &d), &v_inv);

        let e = matrix_exp(&m).unwrap();
        let exp_d = [[libm::exp(l1), 0.0], [0.0, libm::exp(l2)]];
        let want = matmul(&matmul(&v, &exp_d), &v_inv);
        assert!(max_abs_diff(&e, &want) < 1e-13);
    }

    /// `e^0 = I`, and `e^A e^{−A} = I` for any `A` — the group identity, which holds regardless of
    /// structure and so catches a sign or ordering error the structured cases could miss.
    #[test]
    fn the_exponential_of_zero_is_the_identity_and_inverses_compose() {
        let zero = [[0.0; 4]; 4];
        assert_eq!(
            max_abs_diff(&matrix_exp(&zero).unwrap(), &identity::<4>()),
            0.0
        );

        let a = [
            [0.1, -0.4, 2.0, 0.0],
            [1.3, 0.0, -0.2, 0.7],
            [0.0, 0.9, -1.1, 0.3],
            [-0.5, 0.2, 0.0, 0.4],
        ];
        let mut neg = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                neg[i][j] = -a[i][j];
            }
        }
        let product = matmul(&matrix_exp(&a).unwrap(), &matrix_exp(&neg).unwrap());
        assert!(max_abs_diff(&product, &identity::<4>()) < 1e-13);
    }

    /// A non-finite input is reported, not propagated as a matrix of NaNs.
    #[test]
    fn a_non_finite_exponential_argument_is_rejected() {
        assert!(matrix_exp(&[[f64::NAN, 0.0], [0.0, 1.0]]).is_none());
        assert!(matrix_exp(&[[f64::INFINITY, 0.0], [0.0, 1.0]]).is_none());
    }

    /// The double integrator, the one case whose discrete process noise everyone quotes:
    /// `Φ = [[1, Δt], [0, 1]]` and `Q_d = q [[Δt³/3, Δt²/2], [Δt²/2, Δt]]`, derived from the defining
    /// integral in `derivations/process_noise.mac`.
    #[test]
    fn van_loan_reproduces_the_double_integrator_closed_form() {
        let (q, dt) = (1e-12, 60.0);
        let f = [[0.0, 1.0], [0.0, 0.0]];
        let qc = [[0.0, 0.0], [0.0, q]];
        let (phi, qd) = van_loan::<2, 4>(&f, &qc, dt).unwrap();

        assert!(max_abs_diff(&phi, &[[1.0, dt], [0.0, 1.0]]) < 1e-13 * dt);
        let want = [
            [q * dt * dt * dt / 3.0, q * dt * dt / 2.0],
            [q * dt * dt / 2.0, q * dt],
        ];
        assert!(max_abs_diff(&qd, &want) < 1e-14 * want[0][0]);
    }

    /// A scalar Gauss–Markov process: `Φ = e^{−Δt/τ}` and `Q_d = (qτ/2)(1 − e^{−2Δt/τ})`, whose
    /// `Δt → ∞` limit `qτ/2` is the stationary variance that fixes `q = 2σ²/τ` in
    /// [`dynamic_model_compensation`].
    #[test]
    fn van_loan_reproduces_the_gauss_markov_closed_form() {
        let (tau, q, dt) = (200.0, 1e-12, 60.0);
        let f = [[-1.0 / tau]];
        let qc = [[q]];
        let (phi, qd) = van_loan::<1, 2>(&f, &qc, dt).unwrap();

        assert!(libm::fabs(phi[0][0] - libm::exp(-dt / tau)) < 1e-15);
        let want = 0.5 * q * tau * (1.0 - libm::exp(-2.0 * dt / tau));
        assert!(libm::fabs(qd[0][0] - want) < 1e-15 * want);

        // The stationary variance, approached rather than asserted. The tolerance is the measured
        // accuracy of this path at `Δt/τ = 200` (1.4e-14 relative), not a hopeful round number.
        let (_, far) = van_loan::<1, 2>(&f, &qc, 200.0 * tau).unwrap();
        assert!(libm::fabs(far[0][0] - 0.5 * q * tau) < 5e-14 * q * tau);
    }

    /// The full single-axis DMC block at `τ = 200 s`, `q = 1e-12`, `Δt = 60 s`, against the closed form
    /// `derivations/process_noise.mac` derives — evaluated to 50 digits in `mpmath` and rounded, *not*
    /// taken from Maxima's own `float()`. At this `Δt/τ` the closed form's alternating-sign sum loses
    /// about thirteen digits (see the module documentation), so Maxima's f64 evaluation of it disagrees
    /// with the true value in `Q_rr` at the 13th digit. These are the entries too long to trust to
    /// memory, which is why they were derived, and too cancellation-prone to trust to `f64`, which is
    /// why they were then evaluated in extended precision.
    #[test]
    fn van_loan_reproduces_the_dmc_closed_form() {
        let (tau, q, dt) = (200.0, 1e-12, 60.0);
        let f = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0 / tau]];
        let qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, q]];
        let (phi, qd) = van_loan::<3, 6>(&f, &qc, dt).unwrap();

        let want_phi = [
            [1.0, 60.0, 1_632.728_827_268_714_7],
            [0.0, 1.0, 51.836_355_863_656_43],
            [0.0, 0.0, 0.740_818_220_681_717_9],
        ];
        let want_q = [
            [
                3.303_985_406_594_049_5e-5,
                1.332_901_711_697_136e-6,
                2.678_972_598_777_139e-8,
            ],
            [
                1.332_901_711_697_136e-6,
                5.784_498_653_138_012_6e-8,
                1.343_503_894_611_814_1e-9,
            ],
            [
                2.678_972_598_777_139e-8,
                1.343_503_894_611_814_1e-9,
                4.511_883_639_059_735_6e-11,
            ],
        ];

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    libm::fabs(phi[i][j] - want_phi[i][j]) <= 1e-12 * libm::fabs(want_phi[i][j])
                );
                let scale = libm::fabs(want_q[i][j]);
                assert!(libm::fabs(qd[i][j] - want_q[i][j]) <= 1e-9 * scale);
            }
        }
    }

    /// `Q_d` must come back symmetric. The construction averages the off-diagonal pair, so this
    /// measures how much asymmetry there was to average — if that ever stops being rounding-sized, the
    /// averaging is hiding something and this test says so.
    #[test]
    fn the_van_loan_asymmetry_is_rounding_sized_before_it_is_averaged() {
        let (tau, q, dt) = (150.0, 4e-11, 30.0);
        let f = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0 / tau]];
        let qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, q]];

        // Re-derive the unaveraged product the way `van_loan` does internally.
        let mut c = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = -f[i][j] * dt;
                c[i][j + 3] = qc[i][j] * dt;
                c[i + 3][j + 3] = f[j][i] * dt;
            }
        }
        let x = matrix_exp::<6>(&c).unwrap();
        let mut phi = [[0.0; 3]; 3];
        let mut upper = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                phi[i][j] = x[j + 3][i + 3];
                upper[i][j] = x[i][j + 3];
            }
        }
        let raw = matmul(&phi, &upper);

        let scale = raw[0][0];
        for i in 0..3 {
            for j in 0..3 {
                assert!(libm::fabs(raw[i][j] - raw[j][i]) < 1e-12 * scale);
            }
        }
    }

    /// SNC is the double integrator per axis, so it must agree with [`van_loan`] applied to that
    /// dynamic — the two paths exist for ergonomics, not because they are different mathematics.
    #[test]
    fn state_noise_compensation_is_the_double_integrator_per_axis() {
        let dt = 45.0;
        let sigma = [3e-9, 1e-8, 5e-9];
        let q = state_noise_compensation(dt, sigma).unwrap();

        for axis in 0..3 {
            let density = sigma[axis] * sigma[axis];
            let f = [[0.0, 1.0], [0.0, 0.0]];
            let qc = [[0.0, 0.0], [0.0, density]];
            let (_, want) = van_loan::<2, 4>(&f, &qc, dt).unwrap();
            let got = [
                [q[axis][axis], q[axis][axis + 3]],
                [q[axis + 3][axis], q[axis + 3][axis + 3]],
            ];
            assert!(max_abs_diff(&got, &want) < 1e-14 * want[0][0]);
        }

        // Axes do not mix: every cross-axis entry is exactly zero, not merely small.
        for i in 0..6 {
            for j in 0..6 {
                if i % 3 != j % 3 {
                    assert_eq!(q[i][j], 0.0);
                }
            }
        }
    }

    /// Rotating by the identity changes nothing, and rotating there and back returns the original —
    /// the two properties that pin the `A Q Aᵀ` convention without needing a reference matrix.
    #[test]
    fn the_position_velocity_rotation_is_an_orthogonal_congruence() {
        let q = state_noise_compensation(60.0, [2e-9, 7e-9, 4e-9]).unwrap();
        let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(max_abs_diff(&rotate_position_velocity(&q, &eye), &q), 0.0);

        // A rotation of 0.7 rad about z, and its transpose.
        let (c, s) = (libm::cos(0.7), libm::sin(0.7));
        let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let rt = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]];
        let round_trip = rotate_position_velocity(&rotate_position_velocity(&q, &r), &rt);
        assert!(max_abs_diff(&round_trip, &q) < 1e-16);

        // An isotropic noise is invariant under any rotation; an anisotropic one is not, so the
        // function is doing something rather than nothing.
        let iso = state_noise_compensation(60.0, [5e-9; 3]).unwrap();
        assert!(max_abs_diff(&rotate_position_velocity(&iso, &r), &iso) < 1e-24);
        assert!(max_abs_diff(&rotate_position_velocity(&q, &r), &q) > 1e-20);
    }

    /// The nine-state DMC decouples into three independent axes, each of which must reproduce the
    /// single-axis block, with `q = 2σ²/τ` supplying the forcing.
    #[test]
    fn dynamic_model_compensation_is_three_independent_axes() {
        let dt = 60.0;
        let tau = [200.0, 400.0, 150.0];
        let sigma = [1e-8, 3e-8, 2e-8];
        let (phi, q) = dynamic_model_compensation(dt, tau, sigma).unwrap();

        for axis in 0..3 {
            let density = 2.0 * sigma[axis] * sigma[axis] / tau[axis];
            let f = [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0 / tau[axis]],
            ];
            let qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, density]];
            let (want_phi, want_q) = van_loan::<3, 6>(&f, &qc, dt).unwrap();

            let idx = [axis, axis + 3, axis + 6];
            for i in 0..3 {
                for j in 0..3 {
                    let got_phi = phi[idx[i]][idx[j]];
                    let got_q = q[idx[i]][idx[j]];
                    assert!(
                        libm::fabs(got_phi - want_phi[i][j])
                            < 1e-12 * (1.0 + libm::fabs(want_phi[i][j]))
                    );
                    assert!(libm::fabs(got_q - want_q[i][j]) < 1e-10 * libm::fabs(want_q[0][0]));
                }
            }
        }

        // The acceleration variance approaches the stationary sigma it was asked for. `Q_ww` is the
        // smallest entry of a block spanning six orders of magnitude, so its relative accuracy is set
        // by the largest entry — the tolerance is that measured behaviour, not an aspiration.
        let (_, far) = dynamic_model_compensation(500.0 * tau[0], tau, sigma).unwrap();
        assert!(libm::fabs(far[6][6] - sigma[0] * sigma[0]) < 1e-9 * sigma[0] * sigma[0]);
    }

    /// Van Loan's accuracy across the whole range of `Δt/τ` a filter might use, measured against the
    /// scalar Gauss–Markov closed form — which is the one entry with no cancellation, so it is a fair
    /// reference at every ratio.
    ///
    /// The point is the *shape*: the error does not blow up with `Δt/τ` the way the block matrix's
    /// `e^{±Δt/τ}` spread might suggest. It stays at rounding level out to `Δt/τ = 200`. What does cost
    /// digits is dynamic range within one block, which the three-state case shows against the same
    /// reference entry.
    #[test]
    fn van_loan_accuracy_degrades_only_with_the_dynamic_range_of_the_block() {
        let (tau, q) = (200.0, 1e-12);

        for ratio in [0.05, 0.3, 1.0, 5.0, 20.0, 200.0] {
            let dt = ratio * tau;
            let (_, qd) = van_loan::<1, 2>(&[[-1.0 / tau]], &[[q]], dt).unwrap();
            let want = 0.5 * q * tau * (1.0 - libm::exp(-2.0 * dt / tau));
            assert!(
                libm::fabs(qd[0][0] - want) / want < 5e-14,
                "scalar Gauss-Markov at dt/tau = {ratio}"
            );
        }

        // The same quantity as the corner of a block whose largest entry is ~10⁶ times bigger: the
        // relative error grows with that ratio, and 1e-12 is where it lands.
        for ratio in [0.05, 1.0, 20.0] {
            let dt = ratio * tau;
            let f = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0 / tau]];
            let qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, q]];
            let (_, qd) = van_loan::<3, 6>(&f, &qc, dt).unwrap();
            let want = 0.5 * q * tau * (1.0 - libm::exp(-2.0 * dt / tau));
            assert!(
                libm::fabs(qd[2][2] - want) / want < 1e-12,
                "three-state Q_ww at dt/tau = {ratio}"
            );
        }
    }

    /// DMC keeps a correlation the white-noise model destroys: over a step short against `τ` the
    /// acceleration is nearly the same acceleration at both ends, so its normalised autocovariance is
    /// close to one — that persistence is the whole reason DMC exists.
    #[test]
    fn the_dmc_acceleration_stays_correlated_across_a_short_step() {
        let tau = [600.0; 3];
        let sigma = [1e-8; 3];
        let (phi, _) = dynamic_model_compensation(30.0, tau, sigma).unwrap();
        assert!(phi[6][6] > 0.95);

        let (phi_long, _) = dynamic_model_compensation(6000.0, tau, sigma).unwrap();
        assert!(phi_long[6][6] < 0.01);
    }

    /// Every precondition is reported as its own variant, because the three failures have three
    /// different causes at the call site.
    #[test]
    fn the_preconditions_are_reported_separately() {
        let f = [[0.0, 1.0], [0.0, 0.0]];
        let qc = [[0.0, 0.0], [0.0, 1e-12]];
        assert_eq!(
            van_loan::<2, 4>(&f, &qc, -1.0),
            Err(ProcessNoiseError::NegativeStep)
        );
        assert_eq!(
            van_loan::<2, 4>(&f, &qc, f64::NAN),
            Err(ProcessNoiseError::NonFinite)
        );
        assert_eq!(
            van_loan::<2, 4>(&[[f64::NAN, 0.0], [0.0, 0.0]], &qc, 1.0),
            Err(ProcessNoiseError::NonFinite)
        );
        assert_eq!(
            dynamic_model_compensation(60.0, [200.0, 0.0, 200.0], [1e-8; 3]),
            Err(ProcessNoiseError::NonPositiveCorrelationTime)
        );
        assert_eq!(
            dynamic_model_compensation(60.0, [200.0; 3], [1e-8, -1e-8, 1e-8]),
            Err(ProcessNoiseError::NegativeSigma)
        );
        assert_eq!(
            state_noise_compensation(-1.0, [1e-9; 3]),
            Err(ProcessNoiseError::NegativeStep)
        );
        assert_eq!(
            state_noise_compensation(60.0, [1e-9, -1.0, 1e-9]),
            Err(ProcessNoiseError::NegativeSigma)
        );
    }

    /// A zero step accumulates no noise. Worth pinning because it is the boundary the guard admits and
    /// the exponential's scaling path handles specially (`‖0‖ = 0`, so no squarings at all).
    #[test]
    fn a_zero_step_accumulates_no_noise() {
        let q = state_noise_compensation(0.0, [1e-8; 3]).unwrap();
        assert!(q.iter().flatten().all(|&e| e == 0.0));

        let (phi, qd) = dynamic_model_compensation(0.0, [200.0; 3], [1e-8; 3]).unwrap();
        assert_eq!(max_abs_diff(&phi, &identity::<9>()), 0.0);
        assert!(qd.iter().flatten().all(|&e| libm::fabs(e) < 1e-30));
    }

    /// The output is a covariance, so it factors — which is what the filters actually consume.
    #[test]
    fn the_discrete_noise_factors_into_the_square_root_the_filters_take() {
        let q = state_noise_compensation(60.0, [1e-8, 2e-8, 5e-9]).unwrap();
        let s = process_noise_sqrt(&q).unwrap();

        let mut reconstructed = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                let mut acc = 0.0;
                for k in 0..6 {
                    acc += s[i][k] * s[j][k];
                }
                reconstructed[i][j] = acc;
            }
        }
        assert!(max_abs_diff(&reconstructed, &q) < 1e-14 * q[0][0]);

        // A direction that receives no noise leaves `Q` singular, and that is reported rather than
        // factored into something wrong.
        let mut singular = q;
        for j in 0..6 {
            singular[2][j] = 0.0;
            singular[j][2] = 0.0;
        }
        assert!(process_noise_sqrt(&singular).is_none());
    }
}
