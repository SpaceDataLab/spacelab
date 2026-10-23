//! Invariant Extended Kalman Filter (IEKF) for attitude on `SO(3)` (Barrau & Bonnabel 2017).
//!
//! Where the [`crate::mekf`] carries a *multiplicative* error whose measurement Jacobian depends on
//! the current estimate, the invariant EKF defines the error *inside the group* — here the world-
//! frame invariant error, corrected as `R̂ ← Exp(ξ) · R̂`. For the attitude kinematics `Ṙ = R [ω]×`
//! (a group-affine system) Barrau & Bonnabel prove the invariant error propagation is **autonomous**:
//! the transition is exactly `Φ = I` (a common body-rate increment leaves the world-frame error
//! `η = R R̂⁻¹` unchanged), and the vector-measurement Jacobian `H = [r]×` — obtained by forming the
//! innovation in the *reference* frame, `R̂⁻¹ b − r` — depends only on the known inertial reference,
//! never on the estimate. Two consequences this module demonstrates in-test:
//!
//! * **Log-linear exactness** — over a noise-free step the invariant error is transported by the
//!   exact adjoint (a rotation), so its magnitude `|ξ|` is conserved to machine precision for *any*
//!   error size, not just to first order (the EKF/MEKF linearization loses this for large errors).
//! * **State-independent gain** — `H = [r]×` cannot manufacture false observability from a bad
//!   estimate, so the filter converges from a large (80°) initial error and the covariance stays
//!   consistent (ensemble NEES ≈ 3).
//!
//! Attitude-only (no gyro bias) to keep the group-affine structure exact and the demonstration
//! clean; the bias-augmented "imperfect IEKF" and an `SE₂(3)` extended-pose navigation filter
//! (attitude + velocity + position) are the documented extensions.

// Diagonal covariance updates read clearest with an explicit index.
#![allow(clippy::needless_range_loop)]

use crate::lie::{
    UnitQuaternion, V3, dot, eye, hat, inv3, mat_add, mat_sub, mat_vec, matmul, sub, transpose,
};
use crate::robust::{self, RobustOutcome, RobustPolicy};

/// An invariant EKF on `SO(3)`: reference attitude with a 3×3 covariance on the world-frame
/// invariant error `ξ` (`R = Exp(ξ) · R̂`).
#[derive(Clone, Copy, Debug)]
pub struct InvariantEkf {
    attitude: UnitQuaternion,
    covariance: [[f64; 3]; 3],
}

impl InvariantEkf {
    /// Build from an initial attitude estimate and a 3×3 error covariance.
    #[must_use]
    pub fn new(attitude: UnitQuaternion, covariance: [[f64; 3]; 3]) -> Self {
        Self {
            attitude,
            covariance,
        }
    }

    /// Current attitude estimate (inertial→body).
    #[must_use]
    pub fn attitude(&self) -> UnitQuaternion {
        self.attitude
    }

    /// Current 3×3 invariant-error covariance.
    #[must_use]
    pub fn covariance(&self) -> [[f64; 3]; 3] {
        self.covariance
    }

    /// Propagate over `dt` from a rate-gyro sample `gyro` (rad/s) with rate white noise `sigma_gyro`.
    /// The world-frame invariant error is autonomous under a body-rate increment, so `Φ = I` exactly.
    pub fn predict(&mut self, gyro: V3, dt: f64, sigma_gyro: f64) {
        // reference attitude: body-frame rate increment on the right (same op the MEKF validated)
        self.attitude = self
            .attitude
            .mul(UnitQuaternion::from_rotation_vector([
                gyro[0] * dt,
                gyro[1] * dt,
                gyro[2] * dt,
            ]))
            .normalized();

        // The world-frame invariant error is autonomous under a body-rate increment — both the true
        // and estimated attitudes rotate by the same amount, leaving `η = R R̂⁻¹` unchanged, so the
        // exact transition is Φ = I. Isotropic gyro noise maps to the same isotropic world-frame Q.
        let q = sigma_gyro * sigma_gyro * dt;
        for i in 0..3 {
            self.covariance[i][i] += q;
        }
    }

    /// Fold in a unit-vector observation: `body_meas` is the measured body-frame direction of the
    /// known inertial reference `inertial_ref`, with per-axis noise `sigma_meas`. The innovation is
    /// taken in the *reference* frame (`R̂⁻¹ b − r`), giving the state-independent Jacobian
    /// `H = [r]×`. Returns `false` (unchanged) if the innovation covariance is singular.
    ///
    /// Every observation is admitted in full; see [`update_robust`](Self::update_robust) for a star
    /// tracker whose identification can be wrong.
    pub fn update(&mut self, body_meas: V3, inertial_ref: V3, sigma_meas: f64) -> bool {
        self.update_robust(body_meas, inertial_ref, sigma_meas, RobustPolicy::Plain)
            .is_some()
    }

    /// [`update`](Self::update) under a [`RobustPolicy`]. `None` means the innovation covariance is
    /// singular; `Some` with [`accepted`](RobustOutcome::accepted) false means the policy rejected the
    /// observation and the state is unchanged.
    ///
    /// The invariant form makes this cheaper than it looks: `H = [r]×` does not depend on the estimate,
    /// so `δ²` is a statement about the *measurement* rather than about how far the filter has already
    /// drifted. That is exactly the property [`ChiSquareGate`](RobustPolicy::ChiSquareGate) lacks in the
    /// EKF case — the lockout runs through `H`'s dependence on a biased estimate — and it is worth
    /// knowing before choosing a policy here.
    pub fn update_robust(
        &mut self,
        body_meas: V3,
        inertial_ref: V3,
        sigma_meas: f64,
        policy: RobustPolicy,
    ) -> Option<RobustOutcome> {
        // bring the measurement back to the reference frame: R̂⁻¹ b
        let b_ref = self.attitude.conjugate().rotate(body_meas);
        let innovation = sub(b_ref, inertial_ref);

        // H = [r]×  — depends only on the reference direction, never on the estimate (the invariant
        // property, and the reason the gain cannot manufacture false observability).
        let h = hat(inertial_ref);

        let r = sigma_meas * sigma_meas;
        let ht = transpose(&h);
        let pht = matmul(&self.covariance, &ht); // 3×3
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
        let k = matmul(&pht, &s_inv); // 3×3 gain
        let xi = mat_vec(&k, innovation);

        // invariant (world-frame) reset: R̂ ← Exp(ξ) · R̂
        self.attitude = UnitQuaternion::from_rotation_vector(xi)
            .mul(self.attitude)
            .normalized();

        // Joseph-form covariance update
        let kh = matmul(&k, &h);
        let imkh = mat_sub(&eye::<3>(), &kh);
        let imkht = transpose(&imkh);
        let term1 = matmul(&matmul(&imkh, &self.covariance), &imkht);
        let kt = transpose(&k);
        let mut krkt = matmul(&k, &kt);
        for row in &mut krkt {
            for e in row {
                *e *= r;
            }
        }
        self.covariance = mat_add(&term1, &krkt);
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
    use crate::lie::{dot, norm};
    use spacelab_testkit::nan_max;

    fn attitude_error(a: UnitQuaternion, b: UnitQuaternion) -> f64 {
        norm(a.conjugate().mul(b).to_rotation_vector())
    }

    fn diag3(a: [f64; 3]) -> [[f64; 3]; 3] {
        [[a[0], 0.0, 0.0], [0.0, a[1], 0.0], [0.0, 0.0, a[2]]]
    }

    const REF1: V3 = [1.0, 0.0, 0.0];
    const REF2: V3 = [0.0, 1.0, 0.0];

    /// Noise-free: a spinning body with exact vector observations converges to the true attitude —
    /// the convention arbiter (a sign error diverges). Started from a large 80° initial error to
    /// exercise the invariant filter's wide basin of convergence.
    /// One predict from a zero prior leaves the process noise alone, so `σ² dt` is readable directly.
    ///
    /// The transition is exactly the identity for this invariant error, so nothing else contributes:
    /// the covariance after the step *is* the noise the caller asked for. σ = 3e-3 over dt = 0.25
    /// separates `σ² dt` = 2.25e-6 from `2σ dt` = 1.5e-3, from `(σ/σ) dt` = 0.25 and from
    /// `σ² + dt` ≈ 0.25 — none of which the convergence tests can see, since they watch where the
    /// filter settles rather than how wide it is on the way.
    #[test]
    fn a_predict_from_a_zero_prior_returns_the_gyro_noise_it_was_given() {
        let (sigma, dt) = (3e-3, 0.25);
        let mut ekf = InvariantEkf::new(UnitQuaternion::identity(), [[0.0; 3]; 3]);
        ekf.predict([0.02, -0.05, 0.01], dt, sigma);

        let cov = ekf.covariance();
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { sigma * sigma * dt } else { 0.0 };
                assert!(
                    (cov[i][j] - want).abs() < 1e-18,
                    "P[{i}][{j}] = {:e}, want {want:e}",
                    cov[i][j]
                );
            }
        }
    }

    #[test]
    fn converges_to_truth_from_large_error() {
        let q_true0 = UnitQuaternion::from_rotation_vector([0.3, -0.2, 0.5]);
        let omega: V3 = [0.02, -0.01, 0.03];
        let dt = 0.1;

        // ~80° initial attitude error
        let q_init = UnitQuaternion::from_rotation_vector([0.8, 0.6, -0.9]).mul(q_true0);
        let mut ekf = InvariantEkf::new(q_init, diag3([1.0, 1.0, 1.0]));

        let mut q_true = q_true0;
        for _ in 0..400 {
            q_true = q_true
                .mul(UnitQuaternion::from_rotation_vector([
                    omega[0] * dt,
                    omega[1] * dt,
                    omega[2] * dt,
                ]))
                .normalized();
            ekf.predict(omega, dt, 1e-6);
            assert!(ekf.update(q_true.rotate(REF1), REF1, 1e-4));
            assert!(ekf.update(q_true.rotate(REF2), REF2, 1e-4));
        }

        let err = attitude_error(ekf.attitude(), q_true);
        std::eprintln!("LIEKF converges (80° init): attitude err {err:.3e} rad");
        assert!(
            err < 1e-3,
            "attitude error {err:.3e} rad (not converged from 80° initial error)"
        );
    }

    /// Barrau–Bonnabel log-linear exactness: over a noise-free group-affine step the invariant
    /// error is transported by the exact adjoint (a rotation), so `|ξ|` is conserved to machine
    /// precision for *any* error size — the property the EKF linearization lacks.
    #[test]
    fn invariant_error_propagation_is_an_exact_isometry() {
        let q_hat = UnitQuaternion::from_rotation_vector([0.2, 0.7, -0.4]);
        let omega: V3 = [0.05, -0.03, 0.11];
        let dt = 0.5;
        let increment =
            UnitQuaternion::from_rotation_vector([omega[0] * dt, omega[1] * dt, omega[2] * dt]);

        for scale in [1e-3, 0.1, 1.0, 2.5] {
            let xi = [0.3 * scale, -0.5 * scale, 0.8 * scale];
            let q = q_hat.mul(UnitQuaternion::from_rotation_vector(xi)); // R = R̂·Exp(ξ)

            // propagate both by the SAME increment (noise-free, group-affine)
            let q_hat_new = q_hat.mul(increment);
            let q_new = q.mul(increment);

            // exact new invariant error ξ' = log(R̂'⁻¹ R')
            let xi_new = q_hat_new.conjugate().mul(q_new).to_rotation_vector();

            let n0 = norm(xi);
            let n1 = norm(xi_new);
            std::eprintln!(
                "LIEKF isometry: |ξ|={n0:.6e} → |ξ'|={n1:.6e}, |Δ|={:.3e}",
                (n1 - n0).abs()
            );
            assert!(
                (n1 - n0).abs() < 1e-12 * (1.0 + n0),
                "|ξ| not conserved at scale {scale}"
            );
        }
    }

    /// The measurement Jacobian `H = −[r]×` depends only on the reference direction, not on the
    /// attitude estimate — the source of the invariant filter's consistency. Two wildly different
    /// estimates yield the identical `H`.
    #[test]
    fn measurement_jacobian_is_state_independent() {
        let r = [0.3, -0.6, 0.74];
        let h_of = |q: UnitQuaternion| {
            let mut ekf = InvariantEkf::new(q, diag3([0.1, 0.1, 0.1]));
            // capture H by probing the innovation's linear response would be indirect; instead assert
            // the update uses hat(r) regardless of q by comparing the gain structure on equal P.
            ekf.update([1.0, 0.0, 0.0], r, 1.0);
            ekf.covariance()
        };
        let p1 = h_of(UnitQuaternion::identity());
        let p2 = h_of(UnitQuaternion::from_rotation_vector([1.2, -0.8, 0.5]));
        // With a state-independent H and equal prior P and R, the posterior covariance is identical
        // regardless of the attitude estimate.
        let mut max = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                max = nan_max(max, (p1[i][j] - p2[i][j]).abs());
            }
        }
        std::eprintln!("LIEKF state-independent H: max |ΔP| across estimates = {max:.3e}");
        assert!(
            max < 1e-15,
            "posterior covariance depends on the estimate ({max:.3e})"
        );
    }

    /// Consistency (ensemble NEES ≈ 3) under measurement noise from a large initial error.
    #[test]
    fn covariance_is_consistent_ensemble_nees() {
        let q_true = UnitQuaternion::from_rotation_vector([0.1, 0.4, -0.2]);
        let dt = 0.25;
        let sigma = 3.0e-3;
        let runs = 24usize;

        let mut nees_sum = 0.0;
        for run in 0..runs {
            let mut ekf = InvariantEkf::new(
                UnitQuaternion::from_rotation_vector([0.25, -0.2, 0.3]).mul(q_true),
                diag3([0.3, 0.3, 0.3]),
            );
            let mut seed = 0x243f_6a88_85a3_08d3u64 ^ (run as u64).wrapping_mul(0x9e3779b97f4a7c15);
            let mut noise = || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((seed >> 32) as u32 as f64 / 4294967296.0) * 2.0 - 1.0) * 1.7320508075688772
            };
            for _ in 0..300 {
                ekf.predict([0.0; 3], dt, 1e-5);
                for &refv in &[REF1, REF2] {
                    let mut m = q_true.rotate(refv);
                    for c in &mut m {
                        *c += sigma * noise();
                    }
                    ekf.update(m, refv, sigma);
                }
            }
            // world-frame invariant error ξ = log(R_true R̂⁻¹) and its covariance → NEES
            let xi = q_true.mul(ekf.attitude().conjugate()).to_rotation_vector();
            let inv = inv3(&ekf.covariance()).unwrap();
            nees_sum += dot(xi, mat_vec(&inv, xi));
        }
        let avg = nees_sum / runs as f64;
        std::eprintln!("LIEKF ensemble NEES over {runs} runs: {avg:.3} (consistent ≈ 3)");
        assert!(avg > 1.5 && avg < 5.0, "NEES {avg:.3} outside [1.5, 5.0]");
    }

    /// The invariant filter's own version of the MEKF's report check, and the only thing that reads
    /// `weight` or `innovation_variance` here. It also pins the direction of the discount: the update
    /// admits the observation with `R·scale`, so `tr(S)/3` must *rise* by `σ²(scale − 1)` — dividing by
    /// the scale instead would lower it.
    #[test]
    fn the_outcome_reports_the_variance_the_discount_actually_admitted() {
        let attitude = UnitQuaternion::from_rotation_vector([-0.15, 0.25, 0.1]);
        let prior = diag3([4e-2, 1e-2, 2.5e-3]);
        let sigma = 3e-3;
        let truth = attitude.rotate(REF1);
        let measured = [truth[0] + 0.02, truth[1] - 0.01, truth[2] + 0.015];

        let mut plain = InvariantEkf::new(attitude, prior);
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

        let mut inflated = InvariantEkf::new(attitude, prior);
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
