//! The extended-pose group `SE₂(3)` and a bias-augmented invariant navigation filter on it.
//!
//! `SE₂(3)` is the state space of inertial navigation: a rotation `R ∈ SO(3)` (body→world), a velocity
//! `v` and a position `p`, i.e. an `SE(3)` with a **second** translation column. Its algebra is the
//! 9-vector `ξ = [φ (rotation); ν_v (velocity); ν_p (position)]`, mapped by `Exp(ξ) = (Exp(φ), V(φ)ν_v,
//! V(φ)ν_p)` with the same SO(3) left Jacobian `V` as [`crate::se3`] — that module's `Exp`/`Log` core is
//! reused verbatim, only with the extra column. The group itself is validated by its axioms
//! (associativity, inverse, `Exp`/`Log` round-trip, and `g Exp(ξ) g⁻¹ = Exp(Ad_g ξ)`).
//!
//! [`NavigationEkf`] is the **bias-augmented right-invariant EKF** (Barrau & Bonnabel 2018) for
//! strapdown INS: it fuses a rate gyro and accelerometer (with estimated biases) against position
//! fixes. The `SE₂(3)` pose part is *group-affine*, so its error transition is trajectory-independent
//! (the constant `[[0,0,0],[[g]×,0,0],[0,I,0]]`); the IMU biases break that structure — the "imperfect"
//! IEKF — so the bias columns of the transition carry the current estimate. Consistency is shown by an
//! ensemble NEES on a simulated trajectory.

// Index loops are clearest for the small fixed-size matrices; single-char names mirror the algebra.
#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]

use crate::lie::{M3, V3, dot, eye, hat, inv3, mat_add, mat_sub, mat_vec, matmul, sub, transpose};
use crate::robust::{self, RobustOutcome, RobustPolicy};
use crate::se3::{left_jacobian, left_jacobian_inv, so3_exp, so3_log};

fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn neg3(a: V3) -> V3 {
    [-a[0], -a[1], -a[2]]
}

/// An extended pose in `SE₂(3)`: rotation `R` (body→world), velocity `v`, position `p`.
#[derive(Clone, Copy, Debug)]
pub struct Se23 {
    rotation: M3,
    velocity: V3,
    position: V3,
}

impl Se23 {
    /// The identity element.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            velocity: [0.0; 3],
            position: [0.0; 3],
        }
    }

    /// Build from rotation, velocity and position (the matrix is trusted to be a proper rotation).
    #[must_use]
    pub fn from_parts(rotation: M3, velocity: V3, position: V3) -> Self {
        Self {
            rotation,
            velocity,
            position,
        }
    }

    /// The rotation matrix `R` (body→world).
    #[must_use]
    pub fn rotation(&self) -> M3 {
        self.rotation
    }

    /// The velocity `v`.
    #[must_use]
    pub fn velocity(&self) -> V3 {
        self.velocity
    }

    /// The position `p`.
    #[must_use]
    pub fn position(&self) -> V3 {
        self.position
    }

    /// The exponential `Exp(ξ)` of `ξ = [φ, ν_v, ν_p]` — `(Exp(φ), V(φ)ν_v, V(φ)ν_p)`.
    #[must_use]
    pub fn exp(phi: V3, nu_v: V3, nu_p: V3) -> Self {
        let j = left_jacobian(phi);
        Self {
            rotation: so3_exp(phi),
            velocity: mat_vec(&j, nu_v),
            position: mat_vec(&j, nu_p),
        }
    }

    /// The logarithm `Log(g) = (φ, ν_v, ν_p)`.
    #[must_use]
    pub fn log(&self) -> (V3, V3, V3) {
        let phi = so3_log(&self.rotation);
        let j_inv = left_jacobian_inv(phi);
        (
            phi,
            mat_vec(&j_inv, self.velocity),
            mat_vec(&j_inv, self.position),
        )
    }

    /// Group composition `self · other`: `(R₁R₂, R₁v₂+v₁, R₁p₂+p₁)`.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            rotation: matmul::<3, 3, 3>(&self.rotation, &other.rotation),
            velocity: add3(mat_vec(&self.rotation, other.velocity), self.velocity),
            position: add3(mat_vec(&self.rotation, other.position), self.position),
        }
    }

    /// The group inverse `(Rᵀ, −Rᵀv, −Rᵀp)`.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let rt = transpose(&self.rotation);
        Self {
            rotation: rt,
            velocity: neg3(mat_vec(&rt, self.velocity)),
            position: neg3(mat_vec(&rt, self.position)),
        }
    }

    /// The 9×9 adjoint `Ad_g` (ordering `[φ; ν_v; ν_p]`): `[[R,0,0],[[v]×R,R,0],[[p]×R,0,R]]`, satisfying
    /// `g Exp(ξ) g⁻¹ = Exp(Ad_g ξ)`.
    #[must_use]
    pub fn adjoint(&self) -> [[f64; 9]; 9] {
        let r = self.rotation;
        let vx_r = matmul::<3, 3, 3>(&hat(self.velocity), &r);
        let px_r = matmul::<3, 3, 3>(&hat(self.position), &r);
        let mut ad = [[0.0; 9]; 9];
        for i in 0..3 {
            for j in 0..3 {
                ad[i][j] = r[i][j];
                ad[i + 3][j] = vx_r[i][j];
                ad[i + 3][j + 3] = r[i][j];
                ad[i + 6][j] = px_r[i][j];
                ad[i + 6][j + 6] = r[i][j];
            }
        }
        ad
    }
}

/// Noise standard deviations for one [`NavigationEkf::predict`] step: rate-gyro and accelerometer
/// white noise, and the gyro-/accel-bias random-walk rates.
#[derive(Clone, Copy, Debug)]
pub struct ImuNoise {
    /// Rate-gyro white-noise σ (rad/s/√Hz).
    pub gyro: f64,
    /// Accelerometer white-noise σ (m/s²/√Hz).
    pub accel: f64,
    /// Gyro-bias random-walk σ (rad/s²/√Hz).
    pub gyro_bias: f64,
    /// Accel-bias random-walk σ (m/s³/√Hz).
    pub accel_bias: f64,
}

/// A bias-augmented right-invariant EKF for strapdown INS on `SE₂(3)` (Barrau & Bonnabel 2018; Hartley
/// et al. 2020). State: an [`Se23`] extended pose plus gyro and accelerometer biases; the 15-D
/// covariance is on the right-invariant error `[δθ, δv, δp, δb_g, δb_a]` (`X = Exp(ξ)·X̂`). The pose
/// block of the error transition is the *trajectory-independent* group-affine `[[0,0,0],[[g]×,0,0],
/// [0,I,0]]`; the bias columns carry the estimate (`−R̂`, `−[v̂]×R̂`, `−[p̂]×R̂`, `−R̂`) — the "imperfect"
/// IEKF. Position fixes update via the state-coupled Jacobian `H = [−[p̂]×, 0, I, 0, 0]`.
#[derive(Clone, Copy, Debug)]
pub struct NavigationEkf {
    pose: Se23,
    gyro_bias: V3,
    accel_bias: V3,
    covariance: [[f64; 15]; 15],
    gravity: V3,
}

impl NavigationEkf {
    /// Build from an initial pose, IMU-bias estimates, the 15×15 error covariance and the world-frame
    /// gravity vector (e.g. `[0, 0, −9.81]`).
    #[must_use]
    pub fn new(
        pose: Se23,
        gyro_bias: V3,
        accel_bias: V3,
        covariance: [[f64; 15]; 15],
        gravity: V3,
    ) -> Self {
        Self {
            pose,
            gyro_bias,
            accel_bias,
            covariance,
            gravity,
        }
    }

    /// The current extended-pose estimate.
    #[must_use]
    pub fn pose(&self) -> Se23 {
        self.pose
    }

    /// The current gyro-bias estimate.
    #[must_use]
    pub fn gyro_bias(&self) -> V3 {
        self.gyro_bias
    }

    /// The current accelerometer-bias estimate.
    #[must_use]
    pub fn accel_bias(&self) -> V3 {
        self.accel_bias
    }

    /// The current 15×15 error covariance.
    #[must_use]
    pub fn covariance(&self) -> [[f64; 15]; 15] {
        self.covariance
    }

    /// Propagate over `dt` from a rate-gyro sample `gyro` (rad/s, body) and accelerometer `accel`
    /// (m/s², body specific force), correcting by the estimated biases. Mean by the strapdown
    /// mechanization, covariance by `Φ = I + A dt` with `A` the right-invariant error transition.
    pub fn predict(&mut self, gyro: V3, accel: V3, dt: f64, noise: ImuNoise) {
        let r = self.pose.rotation();
        let v = self.pose.velocity();
        let p = self.pose.position();
        let omega = sub(gyro, self.gyro_bias);
        let force = sub(accel, self.accel_bias);
        let a_world = add3(mat_vec(&r, force), self.gravity);

        // Mean: R·Exp(ω dt), v + a_world dt, p + v dt + ½ a_world dt² (world frame).
        let r_new = matmul::<3, 3, 3>(&r, &so3_exp([omega[0] * dt, omega[1] * dt, omega[2] * dt]));
        let v_new = add3(v, scale3(a_world, dt));
        let p_new = add3(add3(p, scale3(v, dt)), scale3(a_world, 0.5 * dt * dt));

        // Right-invariant error transition A (15×15), blocks over [θ, v, p, b_g, b_a].
        let gx = hat(self.gravity);
        let vx_r = matmul::<3, 3, 3>(&hat(v), &r);
        let px_r = matmul::<3, 3, 3>(&hat(p), &r);
        let mut a = [[0.0f64; 15]; 15];
        for i in 0..3 {
            for j in 0..3 {
                a[3 + i][j] = gx[i][j]; // v̇ ← [g]× θ
                a[i][9 + j] = -r[i][j]; // θ̇ ← −R̂ b_g
                a[3 + i][9 + j] = -vx_r[i][j]; // v̇ ← −[v̂]×R̂ b_g
                a[3 + i][12 + j] = -r[i][j]; // v̇ ← −R̂ b_a
                a[6 + i][9 + j] = -px_r[i][j]; // ṗ ← −[p̂]×R̂ b_g
            }
            a[6 + i][3 + i] = 1.0; // ṗ ← v
        }
        // Φ = I + A dt.
        let mut phi = eye::<15>();
        for i in 0..15 {
            for j in 0..15 {
                phi[i][j] += a[i][j] * dt;
            }
        }
        let phit = transpose::<15, 15>(&phi);
        let php = matmul::<15, 15, 15>(&phi, &self.covariance);
        let mut cov = matmul::<15, 15, 15>(&php, &phit);

        // Process noise Q dt (isotropic body noise maps to isotropic world noise).
        let q = [
            noise.gyro * noise.gyro * dt,
            noise.accel * noise.accel * dt,
            0.0,
            noise.gyro_bias * noise.gyro_bias * dt,
            noise.accel_bias * noise.accel_bias * dt,
        ];
        for block in 0..5 {
            for i in 0..3 {
                cov[block * 3 + i][block * 3 + i] += q[block];
            }
        }

        self.covariance = cov;
        self.pose = Se23::from_parts(r_new, v_new, p_new);
    }

    /// Fold in a world-frame position fix `measured` (m) with per-axis noise `sigma`. Innovation
    /// `y − p̂`, Jacobian `H = [−[p̂]×, 0, I, 0, 0]`. Returns `false` if the innovation covariance is
    /// singular.
    pub fn update_position(&mut self, measured: V3, sigma: f64) -> bool {
        self.update_position_robust(measured, sigma, RobustPolicy::Plain)
            .is_some()
    }

    /// [`update_position`](Self::update_position) under a [`RobustPolicy`] — the entry point for a GNSS
    /// fix that can carry a multipath spike. `None` on a singular innovation covariance; `Some` with
    /// [`accepted`](RobustOutcome::accepted) false when the policy rejected it.
    pub fn update_position_robust(
        &mut self,
        measured: V3,
        sigma: f64,
        policy: RobustPolicy,
    ) -> Option<RobustOutcome> {
        let p = self.pose.position();
        let mut h = [[0.0f64; 15]; 3];
        let px = hat(p);
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] = -px[i][j];
            }
            h[i][6 + i] = 1.0;
        }
        self.apply_update(&h, sub(measured, p), sigma, policy)
    }

    /// Fold in a world-frame velocity fix `measured` (m/s) with per-axis noise `sigma`. Innovation
    /// `y − v̂`, Jacobian `H = [−[v̂]×, I, 0, 0, 0]`. Velocity directly observes the accelerometer bias,
    /// which a position-only fix leaves weakly observable. Returns `false` on a singular innovation.
    pub fn update_velocity(&mut self, measured: V3, sigma: f64) -> bool {
        self.update_velocity_robust(measured, sigma, RobustPolicy::Plain)
            .is_some()
    }

    /// [`update_velocity`](Self::update_velocity) under a [`RobustPolicy`]. Same contract as
    /// [`update_position_robust`](Self::update_position_robust).
    pub fn update_velocity_robust(
        &mut self,
        measured: V3,
        sigma: f64,
        policy: RobustPolicy,
    ) -> Option<RobustOutcome> {
        let v = self.pose.velocity();
        let mut h = [[0.0f64; 15]; 3];
        let vx = hat(v);
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] = -vx[i][j];
            }
            h[i][3 + i] = 1.0;
        }
        self.apply_update(&h, sub(measured, v), sigma, policy)
    }

    /// Fold in a body-frame direction measurement `body_meas` of a known world-frame reference
    /// `world_ref` — a magnetometer sensing the geomagnetic field, or a star/sun sensor. Predicted
    /// `ŷ = R̂ᵀ r_w`, innovation `y − ŷ`, Jacobian `H = [R̂ᵀ[r_w]×, 0, 0, 0, 0]` (from `R = Exp(δθ)R̂`,
    /// `Rᵀr_w ≈ R̂ᵀr_w + R̂ᵀ[r_w]×δθ`). Directly observing attitude breaks the tilt↔accel-bias
    /// degeneracy — a velocity residual can no longer be explained away as a tilt — so the
    /// accelerometer bias, only weakly observable under position/velocity alone, becomes strongly
    /// observable. Two non-parallel references pin all three attitude axes. Returns `false` on a
    /// singular innovation covariance.
    pub fn update_attitude_vector(&mut self, body_meas: V3, world_ref: V3, sigma: f64) -> bool {
        self.update_attitude_vector_robust(body_meas, world_ref, sigma, RobustPolicy::Plain)
            .is_some()
    }

    /// [`update_attitude_vector`](Self::update_attitude_vector) under a [`RobustPolicy`] — the entry
    /// point for a magnetometer near a current loop or a star tracker that can mis-identify. Same
    /// contract as [`update_position_robust`](Self::update_position_robust).
    ///
    /// **Tally each stream separately.** This filter takes three, and a pooled
    /// [`RobustTally`](crate::RobustTally) hides a lockout on one of them behind the other two.
    pub fn update_attitude_vector_robust(
        &mut self,
        body_meas: V3,
        world_ref: V3,
        sigma: f64,
        policy: RobustPolicy,
    ) -> Option<RobustOutcome> {
        let rt = transpose(&self.pose.rotation());
        let predicted = mat_vec(&rt, world_ref); // ŷ = R̂ᵀ r_w
        let h_theta = matmul::<3, 3, 3>(&rt, &hat(world_ref)); // R̂ᵀ [r_w]×
        let mut h = [[0.0f64; 15]; 3];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] = h_theta[i][j];
            }
        }
        self.apply_update(&h, sub(body_meas, predicted), sigma, policy)
    }

    /// Shared 3-vector measurement update: innovation covariance `S = H P Hᵀ + σ²I`, gain
    /// `K = P Hᵀ S⁻¹`, right-invariant pose reset `X̂ ← Exp(δξ_pose)·X̂` with additive bias correction,
    /// and a Joseph-form covariance update. `policy` decides what `σ²` the update actually runs with.
    fn apply_update(
        &mut self,
        h: &[[f64; 15]; 3],
        innovation: V3,
        sigma: f64,
        policy: RobustPolicy,
    ) -> Option<RobustOutcome> {
        let ht = transpose::<3, 15>(h); // 15×3
        let pht = matmul::<15, 15, 3>(&self.covariance, &ht); // 15×3
        let hpht = matmul::<3, 15, 3>(h, &pht); // 3×3, before the noise

        let mut nominal = hpht;
        for i in 0..3 {
            nominal[i][i] += sigma * sigma;
        }
        let normalized = dot(innovation, mat_vec(&inv3(&nominal)?, innovation));
        let Some(scale) = policy.variance_scale(normalized, 3) else {
            return Some(robust::rejected(normalized));
        };
        let noise = sigma * sigma * scale;

        let mut s = hpht;
        for i in 0..3 {
            s[i][i] += noise;
        }
        let s_inv = inv3(&s)?;
        let k = matmul::<15, 3, 3>(&pht, &s_inv); // 15×3 gain

        let mut dxi = [0.0f64; 15];
        for i in 0..15 {
            for j in 0..3 {
                dxi[i] += k[i][j] * innovation[j];
            }
        }
        self.pose = Se23::exp(
            [dxi[0], dxi[1], dxi[2]],
            [dxi[3], dxi[4], dxi[5]],
            [dxi[6], dxi[7], dxi[8]],
        )
        .compose(&self.pose);
        for i in 0..3 {
            self.gyro_bias[i] += dxi[9 + i];
            self.accel_bias[i] += dxi[12 + i];
        }

        let kh = matmul::<15, 3, 15>(&k, h);
        let imkh = mat_sub::<15, 15>(&eye::<15>(), &kh);
        let imkht = transpose::<15, 15>(&imkh);
        let term1 = matmul::<15, 15, 15>(&matmul::<15, 15, 15>(&imkh, &self.covariance), &imkht);
        let kt = transpose::<15, 3>(&k);
        let mut krkt = matmul::<15, 3, 15>(&k, &kt);
        for row in &mut krkt {
            for e in row {
                *e *= noise;
            }
        }
        self.covariance = mat_add::<15, 15>(&term1, &krkt);
        Some(RobustOutcome {
            accepted: true,
            weight: 1.0 / scale,
            nominal_normalized_squared: normalized,
            innovation_variance: (s[0][0] + s[1][1] + s[2][2]) / 3.0,
        })
    }
}

fn scale3(v: V3, s: f64) -> V3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[cfg(test)]
mod tests {
    // The equalities are between numbers produced by the same expression, not rounded ones.
    #![allow(clippy::similar_names, clippy::unwrap_used, clippy::float_cmp)]

    use super::*;

    fn max_v3(a: V3, b: V3) -> f64 {
        (a[0] - b[0])
            .abs()
            .max((a[1] - b[1]).abs())
            .max((a[2] - b[2]).abs())
    }

    fn max_se23(a: &Se23, b: &Se23) -> f64 {
        let mut m = max_v3(a.velocity, b.velocity).max(max_v3(a.position, b.position));
        for i in 0..3 {
            for j in 0..3 {
                m = m.max((a.rotation[i][j] - b.rotation[i][j]).abs());
            }
        }
        m
    }

    // Twists ξ = (φ, ν_v, ν_p), incl. a large rotation and the near-identity path.
    const XI: [([f64; 3], [f64; 3], [f64; 3]); 4] = [
        ([0.3, -0.4, 0.2], [1.0, -2.0, 0.5], [-0.7, 3.0, 1.1]),
        ([1.5, 0.9, -1.2], [-0.7, 0.1, 2.0], [2.2, -1.0, 0.3]),
        ([0.01, -0.02, 0.015], [10.0, -5.0, 3.0], [4.0, 2.0, -6.0]),
        ([2.6, -1.1, 0.4], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    ];

    #[test]
    fn exp_log_round_trip() {
        let mut max = 0.0f64;
        for (phi, nv, np) in XI {
            let (phi2, nv2, np2) = Se23::exp(phi, nv, np).log();
            max = max
                .max(max_v3(phi, phi2))
                .max(max_v3(nv, nv2))
                .max(max_v3(np, np2));
        }
        assert!(max < 1e-11, "Exp/Log round-trip error {max:.3e}");
    }

    #[test]
    fn composition_is_associative_and_invertible() {
        let a = Se23::exp(XI[0].0, XI[0].1, XI[0].2);
        let b = Se23::exp(XI[1].0, XI[1].1, XI[1].2);
        let c = Se23::exp(XI[2].0, XI[2].1, XI[2].2);
        let ab_c = a.compose(&b).compose(&c);
        let a_bc = a.compose(&b.compose(&c));
        assert!(max_se23(&ab_c, &a_bc) < 1e-12, "not associative");
        let id = a.compose(&a.inverse());
        assert!(max_se23(&id, &Se23::identity()) < 1e-12, "inverse wrong");
    }

    /// The defining adjoint identity `g Exp(ξ) g⁻¹ = Exp(Ad_g ξ)`.
    #[test]
    fn adjoint_matches_conjugation() {
        let g = Se23::exp(XI[1].0, XI[1].1, XI[1].2);
        let (phi, nv, np) = XI[0];
        let xi = [
            phi[0], phi[1], phi[2], nv[0], nv[1], nv[2], np[0], np[1], np[2],
        ];
        let conj = g.compose(&Se23::exp(phi, nv, np)).compose(&g.inverse());

        let ad = g.adjoint();
        let mut adxi = [0.0; 9];
        for i in 0..9 {
            for j in 0..9 {
                adxi[i] += ad[i][j] * xi[j];
            }
        }
        let rhs = Se23::exp(
            [adxi[0], adxi[1], adxi[2]],
            [adxi[3], adxi[4], adxi[5]],
            [adxi[6], adxi[7], adxi[8]],
        );
        assert!(
            max_se23(&conj, &rhs) < 1e-11,
            "Ad_g does not match conjugation"
        );
    }

    fn diag15(d: [f64; 15]) -> [[f64; 15]; 15] {
        let mut m = [[0.0; 15]; 15];
        for i in 0..15 {
            m[i][i] = d[i];
        }
        m
    }

    fn propagate_truth(r: M3, v: V3, p: V3, omega: V3, force: V3, g: V3, dt: f64) -> (M3, V3, V3) {
        let a_world = [
            mat_vec(&r, force)[0] + g[0],
            mat_vec(&r, force)[1] + g[1],
            mat_vec(&r, force)[2] + g[2],
        ];
        let r_new = matmul::<3, 3, 3>(&r, &so3_exp([omega[0] * dt, omega[1] * dt, omega[2] * dt]));
        let v_new = [
            v[0] + a_world[0] * dt,
            v[1] + a_world[1] * dt,
            v[2] + a_world[2] * dt,
        ];
        let p_new = [
            p[0] + v[0] * dt + 0.5 * a_world[0] * dt * dt,
            p[1] + v[1] * dt + 0.5 * a_world[1] * dt * dt,
            p[2] + v[2] * dt + 0.5 * a_world[2] * dt * dt,
        ];
        (r_new, v_new, p_new)
    }

    /// The process noise the predict adds is the one the caller specified, on the block it names.
    ///
    /// From a zero prior the transition term `Φ P Φᵀ` vanishes, so one step returns `Q dt` alone and
    /// each of the five 3×3 blocks is readable in isolation. The convergence tests below cannot see
    /// this: a mis-scaled or mis-placed process noise changes how fast a noise-free filter tightens,
    /// not where it converges to.
    #[test]
    fn a_predict_from_a_zero_prior_returns_the_process_noise_it_was_given() {
        let noise = ImuNoise {
            gyro: 3e-3,
            accel: 7e-2,
            gyro_bias: 1e-4,
            accel_bias: 5e-4,
        };
        let dt = 0.25;
        let mut ekf = NavigationEkf::new(
            Se23::from_parts(eye::<3>(), [10.0, -4.0, 1.0], [100.0, 20.0, -30.0]),
            [1e-3, -2e-3, 5e-4],
            [1e-2, 3e-3, -4e-3],
            [[0.0; 15]; 15],
            [0.0, 0.0, -9.81],
        );
        ekf.predict([0.02, -0.05, 0.01], [0.3, -0.2, 9.9], dt, noise);

        let want = [
            noise.gyro * noise.gyro * dt,
            noise.accel * noise.accel * dt,
            0.0,
            noise.gyro_bias * noise.gyro_bias * dt,
            noise.accel_bias * noise.accel_bias * dt,
        ];
        let cov = ekf.covariance();
        for i in 0..15 {
            for j in 0..15 {
                let expect = if i == j { want[i / 3] } else { 0.0 };
                assert!(
                    (cov[i][j] - expect).abs() < 1e-18,
                    "P[{i}][{j}] = {:e}, want {expect:e}",
                    cov[i][j]
                );
            }
        }
        assert!(
            want[0] > 0.0 && want[1] > 0.0 && want[3] > 0.0 && want[4] > 0.0,
            "the fixture must have noise to place"
        );
    }

    /// A velocity fix at rest reduces to the scalar Kalman update, with a gain in closed form.
    ///
    /// At zero velocity `[v̂]×` vanishes, so `H = [0, I, 0, 0, 0]` picks the velocity block alone and
    /// a diagonal prior keeps it decoupled: with prior variance `p` and noise `σ`, the gain is
    /// `p/(p + σ²)` and the posterior variance `p σ²/(p + σ²)`, both exactly. Chosen so that `p = 3`
    /// and `σ = 3` separate `σ²` from `2σ` and from `−σ²` — the three ways the innovation covariance
    /// can be built — at 0.25, 1/3 and −0.5.
    #[test]
    fn a_velocity_fix_at_rest_is_the_scalar_kalman_update() {
        let (p, sigma) = (3.0, 3.0);
        let mut prior = [[0.0; 15]; 15];
        for i in 0..15 {
            prior[i][i] = if (3..6).contains(&i) { p } else { 1e-6 };
        }
        let mut ekf = NavigationEkf::new(
            Se23::from_parts(eye::<3>(), [0.0; 3], [0.0; 3]),
            [0.0; 3],
            [0.0; 3],
            prior,
            [0.0, 0.0, -9.81],
        );
        let measured = [2.0, -6.0, 4.0];
        assert!(ekf.update_velocity(measured, sigma), "update must succeed");

        let gain = p / (p + sigma * sigma);
        let v = ekf.pose().velocity();
        for k in 0..3 {
            let want = gain * measured[k];
            assert!(
                (v[k] - want).abs() < 1e-12,
                "velocity[{k}] = {}, want {want}",
                v[k]
            );
        }
        let posterior = p * sigma * sigma / (p + sigma * sigma);
        for i in 3..6 {
            assert!(
                (ekf.covariance()[i][i] - posterior).abs() < 1e-12,
                "P[{i}][{i}] = {}, want {posterior}",
                ekf.covariance()[i][i]
            );
        }
    }

    /// A biased strapdown IMU + 1 Hz position fixes: the bias-augmented RIEKF must converge and
    /// **recover both IMU biases** it was never told, from position alone — the convention arbiter (a
    /// sign error in the transition or the reset diverges). Noise-free so convergence is exact.
    #[test]
    fn ins_gps_converges_and_recovers_biases() {
        use crate::lie::norm;
        let g = [0.0, 0.0, -9.81];
        // Slow roll/pitch/yaw: within a fix interval the accelerometer bias accumulates coherently in
        // velocity (so a fast tumble that averages ∫R·b_a dt to zero would HIDE it), while across the
        // run the attitude reorients enough to decorrelate the body-fixed bias from a tilt error.
        let omega_true = [0.008, -0.006, 0.012];
        let force_true = [1.0, 0.0, 9.81]; // forward push + hover specific force
        let bg_true = [0.01, -0.005, 0.008];
        let ba_true = [0.05, -0.03, 0.02];
        let dt = 0.01;

        let mut r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut v = [10.0, 0.0, 0.0];
        let mut p = [0.0, 0.0, 0.0];

        let cov = diag15([
            0.0025, 0.0025, 0.0025, // attitude (0.05 rad)
            0.25, 0.25, 0.25, // velocity (0.5 m/s)
            25.0, 25.0, 25.0, // position (5 m)
            0.0025, 0.0025, 0.0025, // gyro bias (0.05)
            0.04, 0.04, 0.04, // accel bias (0.2)
        ]);
        let mut ekf = NavigationEkf::new(Se23::from_parts(r, v, p), [0.0; 3], [0.0; 3], cov, g);
        let noise = ImuNoise {
            gyro: 1e-4,
            accel: 1e-3,
            gyro_bias: 1e-5,
            accel_bias: 1e-5,
        };

        let ba_var0 = cov[12][12] + cov[13][13] + cov[14][14]; // initial accel-bias covariance trace
        for step in 0..20000 {
            let (r2, v2, p2) = propagate_truth(r, v, p, omega_true, force_true, g, dt);
            r = r2;
            v = v2;
            p = p2;
            ekf.predict(
                add3(omega_true, bg_true),
                add3(force_true, ba_true),
                dt,
                noise,
            );
            if step % 100 == 99 {
                assert!(ekf.update_position(p, 0.1)); // 1 Hz position fix
                assert!(ekf.update_velocity(v, 0.05)); // and velocity — observes the accel bias
            }
        }
        let pos_err = norm(sub(ekf.pose().position(), p));
        let bg_err = norm(sub(ekf.gyro_bias(), bg_true));
        let cov = ekf.covariance();
        let ba_var1 = cov[12][12] + cov[13][13] + cov[14][14];
        let ba_err0 = norm(ba_true); // initial estimate is zero, so this is the initial error
        let ba_err = norm(sub(ekf.accel_bias(), ba_true));
        std::eprintln!(
            "SE2(3) INS: pos {pos_err:.3e} m, gyro-bias {bg_err:.3e}, accel-bias {ba_err:.3e} \
             (from {ba_err0:.3e}); accel-bias cov trace {ba_var0:.3e} → {ba_var1:.3e}"
        );
        // Group-affine states (pose) and the gyro bias are strongly observable → recovered tightly.
        assert!(pos_err < 1e-2, "position not converged: {pos_err}");
        assert!(bg_err < 5e-3, "gyro bias not recovered: {bg_err}");
        // The accelerometer bias is only weakly observable through the propagation cross-covariance
        // (it is absent from the position/velocity Jacobians), so the demonstration is that the filter
        // *observes* it — its covariance collapses and the estimate moves toward truth — not that it is
        // recovered as fast as the group-affine states.
        assert!(
            ba_var1 < 0.25 * ba_var0,
            "accel-bias covariance did not shrink (not observed)"
        );
        assert!(ba_err < ba_err0, "accel-bias estimate did not improve");
    }

    /// The same biased IMU as above, but now with direct attitude sensing — two non-parallel
    /// world references (a magnetometer's geomagnetic direction and a star/sun sensor) measured in
    /// the body frame. Pinning attitude removes the tilt↔accel-bias ambiguity that made the
    /// accelerometer bias only weakly observable under position/velocity alone, so it is now
    /// **fully** recovered, as tightly as the group-affine states. This is the "specific
    /// measurements" resolution of the weak-observability caveat, not a filter change.
    #[test]
    fn ins_gps_attitude_recovers_accel_bias_fully() {
        use crate::lie::norm;
        let g = [0.0, 0.0, -9.81];
        let omega_true = [0.008, -0.006, 0.012];
        let force_true = [1.0, 0.0, 9.81];
        let bg_true = [0.01, -0.005, 0.008];
        let ba_true = [0.05, -0.03, 0.02];
        let dt = 0.01;

        let mut r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut v = [10.0, 0.0, 0.0];
        let mut p = [0.0, 0.0, 0.0];

        // Two non-parallel world references sensed in the body frame: a geomagnetic-field direction
        // (with a realistic dip) and a horizontal reference — together they fix all three attitude
        // axes. Not unit-normalized on purpose; the Jacobian carries no such assumption.
        let ref_a = [0.42, 0.0, 0.91];
        let ref_b = [0.0, 1.0, 0.0];

        let cov = diag15([
            0.0025, 0.0025, 0.0025, 0.25, 0.25, 0.25, 25.0, 25.0, 25.0, 0.0025, 0.0025, 0.0025,
            0.04, 0.04, 0.04,
        ]);
        let mut ekf = NavigationEkf::new(Se23::from_parts(r, v, p), [0.0; 3], [0.0; 3], cov, g);
        let noise = ImuNoise {
            gyro: 1e-4,
            accel: 1e-3,
            gyro_bias: 1e-5,
            accel_bias: 1e-5,
        };

        for step in 0..20000 {
            let (r2, v2, p2) = propagate_truth(r, v, p, omega_true, force_true, g, dt);
            r = r2;
            v = v2;
            p = p2;
            ekf.predict(
                add3(omega_true, bg_true),
                add3(force_true, ba_true),
                dt,
                noise,
            );
            if step % 100 == 99 {
                let rt = transpose(&r);
                assert!(ekf.update_attitude_vector(mat_vec(&rt, ref_a), ref_a, 1e-3));
                assert!(ekf.update_attitude_vector(mat_vec(&rt, ref_b), ref_b, 1e-3));
                assert!(ekf.update_position(p, 0.1));
                assert!(ekf.update_velocity(v, 0.05));
            }
        }
        let pos_err = norm(sub(ekf.pose().position(), p));
        let bg_err = norm(sub(ekf.gyro_bias(), bg_true));
        let ba_err = norm(sub(ekf.accel_bias(), ba_true));
        std::eprintln!(
            "SE2(3) INS + attitude: pos {pos_err:.3e} m, gyro-bias {bg_err:.3e}, \
             accel-bias {ba_err:.3e}"
        );
        assert!(pos_err < 1e-2, "position not converged: {pos_err}");
        assert!(bg_err < 5e-3, "gyro bias not recovered: {bg_err}");
        // The payoff: with attitude pinned, the accel bias is recovered to the same tight tolerance
        // as the strongly-observable states — full recovery, not just observability.
        assert!(ba_err < 5e-3, "accel bias not fully recovered: {ba_err}");
    }

    /// At the origin the position Jacobian's attitude block `−[p̂]×` vanishes, so with an isotropic
    /// prior `H P Hᵀ = c·I` and the nominal residual has a closed form: `δ² = |ν|²/(c + σ²)`. That pins
    /// `σ` entering the innovation covariance as a *variance* — `σ + σ` and `σ/σ` are numerically
    /// distinguishable from `σ²` at `σ = 3`, and only the closed form says which one is there.
    #[test]
    fn the_nominal_residual_takes_the_square_of_the_measurement_noise() {
        let prior = 2.0;
        let sigma = 3.0;
        let innovation: V3 = [0.6, -0.4, 0.2];

        let mut filter = NavigationEkf::new(
            Se23::identity(),
            [0.0; 3],
            [0.0; 3],
            diag15([prior; 15]),
            [0.0, 0.0, -9.80665],
        );
        let out = filter
            .update_position_robust(innovation, sigma, RobustPolicy::Plain)
            .unwrap();

        let expected = (innovation[0] * innovation[0]
            + innovation[1] * innovation[1]
            + innovation[2] * innovation[2])
            / (prior + sigma * sigma);
        assert!(
            (out.nominal_normalized_squared - expected).abs() < 1e-13 * expected,
            "reported {:.17e}, expected {expected:.17e}",
            out.nominal_normalized_squared
        );
    }

    /// The navigation filter's version of the MEKF report check: the same fix from the same prior, once
    /// undiscounted and once inflated, must differ in `tr(S)/3` by exactly the extra noise variance the
    /// policy admitted. The prior is anisotropic so a mutated trace cannot pass by symmetry.
    #[test]
    fn the_outcome_reports_the_variance_the_discount_actually_admitted() {
        let mut prior = diag15([1e-2; 15]);
        prior[6][6] = 4e-2;
        prior[8][8] = 2.5e-3;
        let sigma = 0.3;
        let measured: V3 = [0.6, -0.4, 0.2];
        let gravity = [0.0, 0.0, -9.80665];

        let mut plain = NavigationEkf::new(Se23::identity(), [0.0; 3], [0.0; 3], prior, gravity);
        let base = plain
            .update_position_robust(measured, sigma, RobustPolicy::Plain)
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

        let mut inflated = NavigationEkf::new(Se23::identity(), [0.0; 3], [0.0; 3], prior, gravity);
        let out = inflated
            .update_position_robust(measured, sigma, policy)
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
