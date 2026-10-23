//! Multiplicative Extended Kalman Filter (MEKF) for spacecraft attitude determination.
//!
//! The industry-standard attitude estimator (Markley & Crassidis, *Fundamentals of Spacecraft
//! Attitude Determination and Control*, §6.2): the reference state is a unit quaternion plus the
//! gyro bias, but the covariance is carried on the **error** state `[δθ (3), δb (3)]` — a minimal,
//! singularity-free 3-parameter attitude error rather than the constrained 4-vector. Each update
//! applies the correction *multiplicatively* (`q̂ ← q̂ ⊗ exp(δθ)`) and resets the error to zero, so
//! the quaternion never leaves the unit sphere and the covariance stays 6×6 and non-degenerate.
//!
//! Attitude is propagated from rate-gyro measurements (`gyro = ω + bias + noise`); vector
//! observations of known inertial directions (star tracker, sun/magnetometer unit vectors) drive
//! the update. The error dynamics are `δθ̇ = −[ω]× δθ − δb`, `δḃ = 0`, with the discrete process
//! noise of the standard gyro model (rate white noise `σ_v`, bias random walk `σ_u`).
//!
//! Convention: quaternion is inertial→body (attitude/passive; see [`crate::lie`]); the body-side
//! multiplicative error `q = q̂ ⊗ δq` gives the clean measurement Jacobian `H_θ = [b̂]×` and the
//! propagation above. The transition's sign is pinned by a second-order-accuracy measurement
//! against the nonlinear step, not by a convention argument.

// Small fixed-size dense linear algebra; index arithmetic is the clearest form here.
#![allow(
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use crate::lie::{
    UnitQuaternion, V3, add, dot, eye, hat, inv3, mat_add, mat_sub, mat_vec, matmul, sub, transpose,
};
use crate::robust::{self, RobustOutcome, RobustPolicy};

/// A Multiplicative EKF: reference quaternion + gyro bias, with a 6×6 error-state covariance
/// `[δθ (rad), δb (rad/s)]`.
#[derive(Clone, Copy, Debug)]
pub struct MultiplicativeEkf {
    attitude: UnitQuaternion,
    bias: V3,
    covariance: [[f64; 6]; 6],
}

impl MultiplicativeEkf {
    /// Build from an initial attitude estimate, gyro-bias estimate, and 6×6 error covariance
    /// `diag([σ_θ² ×3, σ_b² ×3])` (or any symmetric positive-definite prior).
    #[must_use]
    pub fn new(attitude: UnitQuaternion, bias: V3, covariance: [[f64; 6]; 6]) -> Self {
        Self {
            attitude,
            bias,
            covariance,
        }
    }

    /// Current attitude estimate (inertial→body).
    #[must_use]
    pub fn attitude(&self) -> UnitQuaternion {
        self.attitude
    }

    /// Current gyro-bias estimate (rad/s).
    #[must_use]
    pub fn bias(&self) -> V3 {
        self.bias
    }

    /// Current 6×6 error-state covariance.
    #[must_use]
    pub fn covariance(&self) -> [[f64; 6]; 6] {
        self.covariance
    }

    /// Propagate over `dt` from a rate-gyro sample `gyro` (rad/s), with gyro rate-noise `sigma_v`
    /// and bias random-walk `sigma_u` (both rad/s^…, the standard gyro model).
    pub fn predict(&mut self, gyro: V3, dt: f64, sigma_v: f64, sigma_u: f64) {
        let omega = sub(gyro, self.bias);

        // reference quaternion: body-frame rate increment on the right
        self.attitude = self
            .attitude
            .mul(UnitQuaternion::from_rotation_vector([
                omega[0] * dt,
                omega[1] * dt,
                omega[2] * dt,
            ]))
            .normalized();

        // error-state transition Φ = I + F·dt, F = [[−[ω]×, −I], [0, 0]]. The sign is measured, not
        // argued: `the_error_transition_is_second_order_only_with_the_negative_rate_term` propagates
        // truth and estimate through the nonlinear step and shows the linear prediction is wrong by
        // O(dt²) this way and by O(dt) the other.
        let mut phi = eye::<6>();
        let hw = hat(omega);
        for i in 0..3 {
            for j in 0..3 {
                phi[i][j] -= hw[i][j] * dt;
            }
            phi[i][3 + i] = -dt;
        }

        // discrete process noise (Crassidis gyro model)
        let sv2 = sigma_v * sigma_v;
        let su2 = sigma_u * sigma_u;
        let q_tt = sv2 * dt + su2 * dt * dt * dt / 3.0;
        let q_tb = -su2 * dt * dt / 2.0;
        let q_bb = su2 * dt;
        let mut q = [[0.0; 6]; 6];
        for i in 0..3 {
            q[i][i] = q_tt;
            q[i][3 + i] = q_tb;
            q[3 + i][i] = q_tb;
            q[3 + i][3 + i] = q_bb;
        }

        let phit = transpose(&phi);
        let ppt = matmul(&matmul(&phi, &self.covariance), &phit);
        self.covariance = mat_add(&ppt, &q);
    }

    /// Fold in a unit-vector observation: `body_meas` is the measured direction in the body frame of
    /// the known inertial reference direction `inertial_ref`, with per-axis measurement noise
    /// `sigma_meas`. Returns `false` (state unchanged) if the innovation covariance is singular.
    ///
    /// Every observation is admitted in full. A wrong star match goes straight into the attitude —
    /// use [`update_robust`](Self::update_robust) where the star identification can be wrong.
    pub fn update(&mut self, body_meas: V3, inertial_ref: V3, sigma_meas: f64) -> bool {
        self.update_robust(body_meas, inertial_ref, sigma_meas, RobustPolicy::Plain)
            .is_some()
    }

    /// [`update`](Self::update) under a [`RobustPolicy`], for a star tracker whose identification can
    /// fail. `None` means the innovation covariance is singular and no update is posable; `Some` with
    /// [`accepted`](RobustOutcome::accepted) false means the policy rejected the observation and the
    /// state is unchanged. Feed the result to a [`RobustTally`](crate::RobustTally), one per sensor.
    ///
    /// The three axes are one 3-vector observation, not three scalars, so the gate is `χ²(3)` and the
    /// noise scale is [`RobustPolicy::variance_scale`] — the one-step form, whose approximation is
    /// documented there.
    pub fn update_robust(
        &mut self,
        body_meas: V3,
        inertial_ref: V3,
        sigma_meas: f64,
        policy: RobustPolicy,
    ) -> Option<RobustOutcome> {
        let b_hat = self.attitude.rotate(inertial_ref);

        // Body-frame multiplicative error q = q̂ ⊗ δq. With this crate's attitude-matrix product
        // order (A(q̂ ⊗ δq) = A(δq) A(q̂)), the vector-measurement sensitivity is H_θ = [b̂]×.
        let hblock = hat(b_hat);
        let mut h = [[0.0; 6]; 3];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] = hblock[i][j];
            }
        }

        let innovation = sub(body_meas, b_hat);
        let r = sigma_meas * sigma_meas;

        let ht = transpose(&h); // 6×3
        let pht = matmul(&self.covariance, &ht); // 6×3
        let hpht = matmul(&h, &pht); // 3×3, before the noise

        let mut nominal = hpht;
        for i in 0..3 {
            nominal[i][i] += r;
        }
        let normalized = dot(innovation, mat_vec(&inv3(&nominal)?, innovation));
        let Some(scale) = policy.variance_scale(normalized, 3) else {
            return Some(robust::rejected(normalized));
        };
        let r = r * scale;

        let mut s = hpht;
        for i in 0..3 {
            s[i][i] += r;
        }
        let s_inv = inv3(&s)?;
        let k = matmul(&pht, &s_inv); // 6×3 gain

        // state correction δx = K·ν
        let mut dx = [0.0; 6];
        for i in 0..6 {
            dx[i] = k[i][0] * innovation[0] + k[i][1] * innovation[1] + k[i][2] * innovation[2];
        }
        let dtheta = [dx[0], dx[1], dx[2]];
        let dbias = [dx[3], dx[4], dx[5]];
        // multiplicative reset (right, body-frame): q̂ ← q̂ ⊗ exp(δθ)
        self.attitude = self
            .attitude
            .mul(UnitQuaternion::from_rotation_vector(dtheta))
            .normalized();
        self.bias = add(self.bias, dbias);

        // Joseph-form covariance update: P ← (I−KH) P (I−KH)ᵀ + K R Kᵀ
        let kh = matmul(&k, &h); // 6×6
        let imkh = mat_sub(&eye::<6>(), &kh);
        let imkht = transpose(&imkh);
        let term1 = matmul(&matmul(&imkh, &self.covariance), &imkht);
        let kt = transpose(&k); // 3×6
        let mut kkt = matmul(&k, &kt); // 6×6
        for i in 0..6 {
            for j in 0..6 {
                kkt[i][j] *= r;
            }
        }
        self.covariance = mat_add(&term1, &kkt);
        Some(RobustOutcome {
            accepted: true,
            weight: 1.0 / scale,
            nominal_normalized_squared: normalized,
            innovation_variance: (s[0][0] + s[1][1] + s[2][2]) / 3.0,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unreadable_literal,
        clippy::unwrap_used,
        clippy::cast_precision_loss,
        clippy::cast_lossless,
        // The equalities are between numbers produced by the same expression, not rounded ones.
        clippy::float_cmp
    )]
    use super::*;
    use crate::lie::{dot, mat_vec, norm, scale};

    /// Angular error (rad) between two attitudes.
    fn attitude_error(a: UnitQuaternion, b: UnitQuaternion) -> f64 {
        norm(a.conjugate().mul(b).to_rotation_vector())
    }

    fn diag6(a: [f64; 6]) -> [[f64; 6]; 6] {
        let mut m = [[0.0; 6]; 6];
        for i in 0..6 {
            m[i][i] = a[i];
        }
        m
    }

    /// Two non-parallel inertial references (e.g. a star and the sun direction).
    const REF1: V3 = [1.0, 0.0, 0.0];
    const REF2: V3 = [0.0, 1.0, 0.0];

    /// Noise-free: exact gyro + exact vector measurements ⇒ the MEKF must converge to the true
    /// attitude and recover the true gyro bias. This is the convention arbiter (a sign error would
    /// diverge). Truth spins at a constant body rate with a constant bias offset on the gyro.
    #[test]
    fn converges_to_truth_and_recovers_bias_noise_free() {
        let q_true0 = UnitQuaternion::from_rotation_vector([0.3, -0.2, 0.5]);
        let bias_true: V3 = [1.0e-3, -2.0e-3, 1.5e-3];
        let omega_true: V3 = [0.02, -0.01, 0.03];
        let dt = 0.1;

        // start 20° off in attitude, zero bias estimate, loose prior
        let q_init = UnitQuaternion::from_rotation_vector([0.2, 0.15, -0.25]).mul(q_true0);
        let mut ekf =
            MultiplicativeEkf::new(q_init, [0.0; 3], diag6([0.2, 0.2, 0.2, 1e-4, 1e-4, 1e-4]));

        let mut q_true = q_true0;
        for _ in 0..600 {
            // truth propagation + gyro (biased) and exact body observations
            q_true = q_true
                .mul(UnitQuaternion::from_rotation_vector([
                    omega_true[0] * dt,
                    omega_true[1] * dt,
                    omega_true[2] * dt,
                ]))
                .normalized();
            let gyro = add(omega_true, bias_true);
            // a tiny process noise keeps the gain alive so the bias converges fully (Q=0 would let
            // P collapse and freeze the estimate short of truth)
            ekf.predict(gyro, dt, 1e-6, 1e-8);
            assert!(ekf.update(q_true.rotate(REF1), REF1, 1e-4));
            assert!(ekf.update(q_true.rotate(REF2), REF2, 1e-4));
        }

        let att_err = attitude_error(ekf.attitude(), q_true);
        let bias_err = norm(sub(ekf.bias(), bias_true));
        std::eprintln!(
            "MEKF noise-free: attitude err {att_err:.3e} rad, bias err {bias_err:.3e} rad/s"
        );
        assert!(
            att_err < 1e-3,
            "attitude error {att_err:.3e} rad (not converged)"
        );
        assert!(
            bias_err < 2e-4,
            "bias error {bias_err:.3e} rad/s (bias not recovered)"
        );
    }

    /// The rate at which the estimate and the truth are spun apart, and the error state that
    /// separates them: `δθ` from the body-side error `q = q̂ ⊗ δq`, `δb` from the gyro bias.
    const OMEGA: V3 = [0.4, -0.7, 0.5];
    const DTHETA: V3 = [6e-7, -4.8e-7, 6.4e-7];
    const DBIAS: V3 = [7e-7, 2e-7, -6e-7];

    /// One step of the reference propagation `predict` performs on the attitude.
    fn spin(q: UnitQuaternion, rate: V3, dt: f64) -> UnitQuaternion {
        q.mul(UnitQuaternion::from_rotation_vector(scale(rate, dt)))
            .normalized()
    }

    /// The error state after one step, linearized: `Φ [δθ; δb]` with `F_θθ = sign·[ω]×`.
    fn transported_error(sign: f64, dt: f64) -> V3 {
        add(
            sub(DTHETA, scale(DBIAS, dt)),
            scale(mat_vec(&hat(OMEGA), DTHETA), sign * dt),
        )
    }

    /// The `[ω]×` sign in the error-state transition, measured against the nonlinear step.
    ///
    /// `Φ = I + F·dt` truncates the exact discrete transition at first order in `dt`, so with the
    /// right `F` the linear prediction of the error after one step is wrong by `O(dt²)` and with the
    /// sign flipped it is wrong by `O(dt)` — the `[ω]× dt` term itself. Halving `dt` separates the
    /// two by inspection: the residual falls by 4 one way and by 2 the other. No convention argument
    /// is involved; truth and estimate are propagated through the same nonlinear step `predict`
    /// uses, differing only in the gyro bias each is given.
    #[test]
    fn the_error_transition_is_second_order_only_with_the_negative_rate_term() {
        let q_hat = UnitQuaternion::from_rotation_vector([0.3, -0.2, 0.5]);
        let q_true = q_hat
            .mul(UnitQuaternion::from_rotation_vector(DTHETA))
            .normalized();
        let mut residuals = [[0.0; 4]; 2];
        for (k, dt) in [0.2, 0.1, 0.05, 0.025].into_iter().enumerate() {
            // the gyro sees the true rate plus the true bias; the filter subtracts its own estimate,
            // so its rate is off by exactly DBIAS and the two attitudes separate over the step
            let omega_hat = add(OMEGA, DBIAS);
            let after = spin(q_hat, omega_hat, dt)
                .conjugate()
                .mul(spin(q_true, OMEGA, dt))
                .to_rotation_vector();
            for (s, sign) in [-1.0, 1.0].into_iter().enumerate() {
                residuals[s][k] = norm(sub(after, transported_error(sign, dt)));
            }
        }
        for k in 1..4 {
            let (neg, pos) = (
                residuals[0][k] / residuals[0][k - 1],
                residuals[1][k] / residuals[1][k - 1],
            );
            assert!(
                neg < 0.30,
                "−[ω]× residual shrank by {neg:.3} on halving dt, not the ≈0.25 of a second-order \
                 truncation: {:?}",
                residuals[0]
            );
            assert!(
                pos > 0.45,
                "+[ω]× residual shrank by {pos:.3} on halving dt — if it is also second order the \
                 test cannot tell the signs apart and proves nothing: {:?}",
                residuals[1]
            );
        }
    }

    /// The covariance carries that transition.
    ///
    /// With a rank-one prior `δδᵀ` and no process noise, one step must give exactly `(Φδ)(Φδ)ᵀ` —
    /// the outer product of the error the previous test measured. That ties `predict`'s covariance
    /// arithmetic to the transition rather than to a second copy of the same matrix.
    #[test]
    fn a_rank_one_prior_propagates_as_the_outer_product_of_the_transported_error() {
        let dt = 0.05;
        let delta = [
            DTHETA[0], DTHETA[1], DTHETA[2], DBIAS[0], DBIAS[1], DBIAS[2],
        ];
        let mut prior = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                prior[i][j] = delta[i] * delta[j];
            }
        }
        let bias: V3 = [1e-3, -2e-3, 1.5e-3];
        let mut ekf = MultiplicativeEkf::new(UnitQuaternion::identity(), bias, prior);
        ekf.predict(add(OMEGA, bias), dt, 0.0, 0.0);

        let theta = transported_error(-1.0, dt);
        let want = [theta[0], theta[1], theta[2], DBIAS[0], DBIAS[1], DBIAS[2]];
        let got = ekf.covariance();
        for i in 0..6 {
            for j in 0..6 {
                let w = want[i] * want[j];
                assert!(
                    (got[i][j] - w).abs() <= 1e-12 * w.abs().max(1e-14),
                    "P[{i}][{j}] = {:.17e}, want {w:.17e}",
                    got[i][j]
                );
            }
        }
    }

    /// The discrete process noise against the integral that defines it.
    ///
    /// `Q = ∫₀^Δt Φ(s) G Q_c Gᵀ Φ(s)ᵀ ds` with `Φ(s) = I + F s` at `ω = 0` — the case the Crassidis
    /// gyro model is derived for — and `G Q_c Gᵀ = diag(σ_v² I, σ_u² I)`. That makes the integrand
    /// blocks `σ_v² + s²σ_u²`, `−s σ_u²` and `σ_u²`, all of degree ≤ 2, on which Simpson is exact:
    /// the quadrature is the closed form's own definition evaluated a different way, not a second
    /// copy of it. A zero prior kills the `Φ P Φᵀ` term, so one step returns `Q` alone.
    #[test]
    fn the_process_noise_is_the_integral_of_the_continuous_model_through_the_transition() {
        let (sigma_v, sigma_u, dt) = (1e-3, 3e-3, 2.0);
        let mut ekf = MultiplicativeEkf::new(UnitQuaternion::identity(), [0.0; 3], [[0.0; 6]; 6]);
        ekf.predict([0.1, -0.2, 0.3], dt, sigma_v, sigma_u);

        let panels = 8_usize;
        let h = dt / panels as f64;
        let (mut tt, mut tb, mut bb) = (0.0, 0.0, 0.0);
        for p in 0..panels {
            for (node, weight) in [(0.0, 1.0), (0.5, 4.0), (1.0, 1.0)] {
                let s = p as f64 * h + node * h;
                let w = weight * h / 6.0;
                tt += w * (sigma_v * sigma_v + s * s * sigma_u * sigma_u);
                tb += w * (-s * sigma_u * sigma_u);
                bb += w * (sigma_u * sigma_u);
            }
        }

        let got = ekf.covariance();
        for i in 0..6 {
            for j in 0..6 {
                let want = match (i, j) {
                    _ if i == j && i < 3 => tt,
                    _ if i == j => bb,
                    _ if j == i + 3 || i == j + 3 => tb,
                    _ => 0.0,
                };
                assert!(
                    (got[i][j] - want).abs() <= 1e-13 * want.abs().max(1e-12),
                    "Q[{i}][{j}] = {:.17e}, want {want:.17e}",
                    got[i][j]
                );
            }
        }
    }

    /// Consistency (ensemble NEES): over many independent noisy runs, the average normalized
    /// estimation error squared `δθᵀ P_θθ⁻¹ δθ` must be ≈ 3 (the attitude dimension) — the standard
    /// filter-consistency test (Bar-Shalom). Far below 3 ⇒ pessimistic; far above ⇒ overconfident.
    /// Deterministic per-run zero-mean pseudo-noise, so the test is reproducible.
    #[test]
    fn covariance_is_consistent_ensemble_nees() {
        let q_true = UnitQuaternion::from_rotation_vector([0.1, 0.4, -0.2]);
        let dt = 0.25;
        let sigma = 3.0e-3;
        let runs = 24usize;

        let mut nees_sum = 0.0;
        for run in 0..runs {
            let mut ekf = MultiplicativeEkf::new(
                UnitQuaternion::from_rotation_vector([0.1, -0.1, 0.15]).mul(q_true),
                [0.0; 3],
                diag6([0.1, 0.1, 0.1, 1e-6, 1e-6, 1e-6]),
            );
            // per-run seed; zero-mean uniform in [−1, 1)
            let mut seed =
                0x9e37_79b9_7f4a_7c15u64 ^ (run as u64).wrapping_mul(0xd1b5_4a32_d192_ed03);
            let mut noise = || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 32) as u32 as f64 / 4294967296.0) * 2.0 - 1.0
            };
            for _ in 0..300 {
                ekf.predict([0.0; 3], dt, 1e-5, 1e-7);
                for &refv in &[REF1, REF2] {
                    let mut m = q_true.rotate(refv);
                    // scale uniform[−1,1) (std 1/√3) to std σ so the injected noise matches the R
                    // the filter is told — otherwise the filter looks pessimistic (NEES < dim)
                    for c in &mut m {
                        *c += sigma * 1.732_050_807_568_877_2 * noise();
                    }
                    ekf.update(m, refv, sigma);
                }
            }
            // body-frame attitude error and its 3×3 covariance block → NEES
            let dtheta = ekf.attitude().conjugate().mul(q_true).to_rotation_vector();
            let p = ekf.covariance();
            let p_theta = [
                [p[0][0], p[0][1], p[0][2]],
                [p[1][0], p[1][1], p[1][2]],
                [p[2][0], p[2][1], p[2][2]],
            ];
            let inv = inv3(&p_theta).unwrap();
            nees_sum += dot(dtheta, mat_vec(&inv, dtheta));
        }
        let avg = nees_sum / runs as f64;
        std::eprintln!("MEKF ensemble NEES over {runs} runs: {avg:.3} (consistent ≈ 3)");
        assert!(
            avg > 1.5 && avg < 5.0,
            "NEES {avg:.3} outside [1.5, 5.0] — inconsistent"
        );
    }

    /// `RobustOutcome`'s three fields are the filter's report on what it did with the observation, and
    /// nothing else in the suite reads them: `weight` is the reciprocal of the policy's variance scale
    /// and `innovation_variance` is `tr(S)/3` *after* the discount. Run the same observation twice from
    /// the same prior, once undiscounted and once inflated: `tr(HPHᵀ)` is common to both, so the
    /// reported variance must move by exactly the extra measurement noise admitted. The prior is
    /// deliberately anisotropic — a mutated trace that drops or reorders a diagonal term would pass on
    /// an isotropic one.
    #[test]
    fn the_outcome_reports_the_variance_the_discount_actually_admitted() {
        let attitude = UnitQuaternion::from_rotation_vector([0.1, -0.2, 0.05]);
        let prior = diag6([4e-2, 1e-2, 2.5e-3, 1e-6, 4e-6, 9e-6]);
        let sigma = 3e-3;
        let truth = attitude.rotate(REF1);
        let measured = [truth[0] + 0.02, truth[1] - 0.01, truth[2] + 0.015];

        let mut plain = MultiplicativeEkf::new(attitude, [0.0; 3], prior);
        let base = plain
            .update_robust(measured, REF1, sigma, RobustPolicy::Plain)
            .unwrap();
        assert!(base.accepted);
        assert_eq!(base.weight, 1.0);

        let policy = RobustPolicy::ChiSquareInflation {
            threshold: base.nominal_normalized_squared / 2.0,
        };
        let scale = policy
            .variance_scale(base.nominal_normalized_squared, 3)
            .unwrap();
        assert!(scale > 1.5, "the control must actually discount: {scale}");

        let mut inflated = MultiplicativeEkf::new(attitude, [0.0; 3], prior);
        let out = inflated
            .update_robust(measured, REF1, sigma, policy)
            .unwrap();

        assert_eq!(
            out.nominal_normalized_squared,
            base.nominal_normalized_squared
        );
        assert_eq!(out.weight, 1.0 / scale);
        let admitted = out.innovation_variance - base.innovation_variance;
        let expected = sigma * sigma * (scale - 1.0);
        assert!(
            (admitted - expected).abs() < 1e-9 * expected,
            "admitted {admitted:.17e}, expected {expected:.17e}"
        );
    }
}
