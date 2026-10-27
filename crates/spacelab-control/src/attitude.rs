//! Attitude control laws — the torque that removes a pointing error.
//!
//! [`spacelab_propagation::attitude_law`] says where to point and
//! [`spacelab_propagation::attitude::integrate_attitude_torqued`] says how the body responds to a
//! moment. This module supplies the third piece: given the error between the two, the moment to
//! command. That moment goes straight into the integrator's `moment` callback, so the closed loop
//! needs no new integrator — the RKMK4 Lie-group step already keeps `RᵀR = I` while the controller
//! drives the error down.
//!
//! # Two laws, and why both
//!
//! [`QuaternionPd`] is Wie's proportional–derivative law, `u = −K q_v − P δω`. It is the one whose
//! small-angle behaviour is an ordinary second-order system, which makes it the law to check an
//! implementation against: the closed loop has a natural frequency and a damping ratio that follow
//! from `K`, `P` and the inertia by algebra, so a simulated response either matches the analytic
//! solution or the code is wrong. There is no tolerance to tune.
//!
//! [`MrpLyapunov`] is the Schaub–Junkins law, `u = −K σ − P δω + ω × Iω`. The gyroscopic term
//! cancels the body's own nonlinearity rather than asking the gains to dominate it, so the law comes
//! with a Lyapunov function whose derivative is negative semi-definite for *any* positive gains and
//! *any* initial error, not just small ones. It is also Basilisk's reference law, which is what
//! makes a like-for-like comparison possible later.
//!
//! # The short way round
//!
//! A rotation has two unit quaternions, `q` and `−q`, and a proportional law does not treat them
//! alike: `u = −K q_v` with `q₀ < 0` commands the *long* way round — up to very nearly a full extra
//! revolution to reach an attitude that was degrees away. This is the unwinding phenomenon, and it
//! is not an implementation slip but a consequence of `SO(3)`'s topology: Bhat and Bernstein (2000)
//! showed no continuous state feedback on it can be globally asymptotically stable. The practical
//! resolution used here is the standard one — choose the representative with `q₀ ≥ 0`, which is the
//! shorter rotation, and accept that the antipodal set is where continuity is given up.
//!
//! That choice does a second job for free. The modified Rodrigues parameters are `σ = q_v/(1 + q₀)`,
//! singular at `q₀ = −1`; keeping `q₀ ≥ 0` bounds `‖σ‖ ≤ 1` and puts the singularity a full
//! revolution away from anything the controller will see.
//!
//! # References
//!
//! - B. Wie and P. M. Barba, *Quaternion feedback for spacecraft large angle manoeuvres*, Journal of
//!   Guidance, Control and Dynamics 8(3), 1985.
//! - H. Schaub and J. L. Junkins, *Analytical Mechanics of Space Systems*, 4th ed., AIAA, §8.4.
//! - S. P. Bhat and D. S. Bernstein, *A topological obstruction to continuous global stabilization
//!   of rotational motion*, Systems & Control Letters 39(1), 2000.
//!
//! Gate: `derivations/attitude_control.mac` derives Shepperd's branches, the body-rate kinematics
//! of the error `[NR]ᵀ[NB]`, the MRP kinematics, `V̇ = −P‖ω‖²` for the Schaub–Junkins law with any
//! symmetric inertia, and the small-angle gains and second-order loop of the PD law;
//! `derivations/attitude_control.gp` re-checks the first four in exact rationals. The module tests
//! `the_error_is_the_body_rotation_relative_to_the_reference` (G1),
//! `the_lyapunov_derivative_along_the_body_kinematics_is_minus_the_rate_damping` (G2),
//! `the_quaternion_of_a_rodrigues_rotation_is_the_half_angle_pair` (G1) and
//! `the_mrp_law_linearises_to_a_quarter_of_its_nominal_stiffness` (G3) tie the code to it.

use spacelab_propagation::attitude::{Inertia, Mat3, Vec3};

use crate::margin::SmallAngleGains;

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

/// `Aᵀ · B`, the composition that expresses "how far `b` is from `a`" for two inertial-from-body
/// rotations sharing an inertial reference: `[NR]ᵀ[NB] = [RB]`.
fn transpose_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[0][i].mul_add(b[0][j], a[1][i].mul_add(b[1][j], a[2][i] * b[2][j]));
        }
    }
    out
}

/// The unit quaternion `(q₀, q_v)` of a rotation matrix, sign-fixed to `q₀ ≥ 0`.
///
/// Branch selection follows Shepperd: the four candidate denominators are `1 ± Mᵢᵢ ± Mⱼⱼ ± M_kk` and
/// taking the largest keeps the division away from zero. The naive `q₀ = √(1 + tr M)/2` branch alone
/// loses all its significant digits at a 180° rotation, where `tr M → −1`.
fn quaternion_of(m: &Mat3) -> [f64; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let candidates = [
        1.0 + trace,
        1.0 + 2.0 * m[0][0] - trace,
        1.0 + 2.0 * m[1][1] - trace,
        1.0 + 2.0 * m[2][2] - trace,
    ];
    let mut best = 0;
    for (i, &c) in candidates.iter().enumerate() {
        if c > candidates[best] {
            best = i;
        }
    }
    let root = libm::sqrt(candidates[best].max(0.0));
    let half = 0.5 / root.max(f64::MIN_POSITIVE);

    let mut q = match best {
        0 => [
            0.5 * root,
            (m[2][1] - m[1][2]) * half,
            (m[0][2] - m[2][0]) * half,
            (m[1][0] - m[0][1]) * half,
        ],
        1 => [
            (m[2][1] - m[1][2]) * half,
            0.5 * root,
            (m[0][1] + m[1][0]) * half,
            (m[0][2] + m[2][0]) * half,
        ],
        2 => [
            (m[0][2] - m[2][0]) * half,
            (m[0][1] + m[1][0]) * half,
            0.5 * root,
            (m[1][2] + m[2][1]) * half,
        ],
        _ => [
            (m[1][0] - m[0][1]) * half,
            (m[0][2] + m[2][0]) * half,
            (m[1][2] + m[2][1]) * half,
            0.5 * root,
        ],
    };
    if q[0] < 0.0 {
        for x in &mut q {
            *x = -*x;
        }
    }
    q
}

/// How far the body is from where it was told to be, and how fast that gap is changing.
///
/// Build it with [`AttitudeError::new`] from the actual and reference orientations. The stored
/// quaternion always has `q₀ ≥ 0` — see the module documentation on unwinding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttitudeError {
    quaternion: [f64; 4],
    rate: Vec3,
    omega: Vec3,
}

impl AttitudeError {
    /// `actual` and `reference` are the inertial-from-body rotations `[NB]` and `[NR]` — the matrix
    /// [`spacelab_propagation::attitude::integrate_attitude_torqued`] evolves by `Ṙ = R[ω]×`, and the
    /// transpose of the inertial→body matrix an
    /// [`spacelab_propagation::attitude_law::AttitudeProvider`] returns. `omega` is the body rate and
    /// `omega_reference` the reference's, both in body axes.
    ///
    /// The error rotation is `[RB] = [NR]ᵀ[NB]`, the body relative to the reference: it is the one
    /// whose kinematics are `Ė = E[ω]×` in the body rate, which is what both laws and their Lyapunov
    /// function assume. `[NB][NR]ᵀ` agrees with it only when the reference is the identity; for any
    /// other reference its axis is the body error rotated by `[NR]`, and a loop closed on it
    /// commands torque about the wrong axis — with a 90° reference and a 60° error it was still
    /// 39° off after 200 s where the right error converges to 0.07°.
    #[must_use]
    pub fn new(actual: &Mat3, reference: &Mat3, omega: Vec3, omega_reference: Vec3) -> Self {
        let error_rotation = transpose_mul(reference, actual);
        Self {
            quaternion: quaternion_of(&error_rotation),
            rate: [
                omega[0] - omega_reference[0],
                omega[1] - omega_reference[1],
                omega[2] - omega_reference[2],
            ],
            omega,
        }
    }

    /// The error quaternion, `(q₀, q₁, q₂, q₃)`, with `q₀ ≥ 0`.
    #[must_use]
    pub const fn quaternion(&self) -> [f64; 4] {
        self.quaternion
    }

    /// The vector part of the error quaternion — what a proportional law acts on.
    #[must_use]
    pub const fn quaternion_vector(&self) -> Vec3 {
        [self.quaternion[1], self.quaternion[2], self.quaternion[3]]
    }

    /// The modified Rodrigues parameters, `σ = q_v/(1 + q₀)`. Bounded by 1 because `q₀ ≥ 0`.
    #[must_use]
    pub fn mrp(&self) -> Vec3 {
        let d = 1.0 + self.quaternion[0];
        [
            self.quaternion[1] / d,
            self.quaternion[2] / d,
            self.quaternion[3] / d,
        ]
    }

    /// The principal rotation angle of the error, in radians, on `[0, π]`.
    #[must_use]
    pub fn angle(&self) -> f64 {
        2.0 * libm::acos(self.quaternion[0].clamp(-1.0, 1.0))
    }

    /// `δω = ω − ω_ref`, body axes.
    #[must_use]
    pub const fn rate(&self) -> Vec3 {
        self.rate
    }

    /// The absolute body rate `ω`, which the gyroscopic cancellation needs.
    #[must_use]
    pub const fn omega(&self) -> Vec3 {
        self.omega
    }
}

/// A feedback law: an attitude error in, a commanded body moment out.
pub trait ControlLaw {
    /// The commanded moment in body axes, N·m.
    fn torque(&self, error: &AttitudeError, inertia: &Inertia) -> Vec3;
}

/// Wie's quaternion proportional–derivative law, `u = −K q_v − P δω`.
///
/// For a principal axis with moment `I` and a small error the closed loop is
/// `I δθ̈ + P δθ̇ + (K/2) δθ = 0`, since `q_v ≈ δθ/2`. So
///
/// ```text
/// ωₙ = √(K / 2I)        ζ = P / √(2 I K)
/// ```
///
/// and [`QuaternionPd::critically_damped`] inverts the second of those. That relation is the
/// module's sharpest test: the simulated response has to match the analytic second-order solution
/// pointwise, not merely settle somewhere reasonable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuaternionPd {
    proportional: f64,
    derivative: f64,
}

impl QuaternionPd {
    /// `None` unless both gains are finite and strictly positive. A non-positive gain is an open
    /// loop or a divergent one, and neither is a controller.
    #[must_use]
    pub fn new(proportional: f64, derivative: f64) -> Option<Self> {
        let ok = |x: f64| x.is_finite() && x > 0.0;
        (ok(proportional) && ok(derivative)).then_some(Self {
            proportional,
            derivative,
        })
    }

    /// The gain pair that puts a principal axis of moment `moment` at `ζ = 1`: `P = √(2 I K)`.
    ///
    /// There is no separate validation of `moment`, because [`Self::new`] already rejects every
    /// value one would reject: `√(2 I K)` is `+inf` for an infinite moment, `NaN` for a negative or
    /// `NaN` one, and `0` for a zero one, and none of the three is a finite positive gain. A guard
    /// here would refuse exactly the same inputs one step earlier, which is a branch no input can
    /// tell apart from its absence.
    #[must_use]
    pub fn critically_damped(proportional: f64, moment: f64) -> Option<Self> {
        Self::new(proportional, libm::sqrt(2.0 * moment * proportional))
    }

    /// The undamped natural frequency this gain gives a principal axis of moment `moment`, rad/s.
    #[must_use]
    pub fn natural_frequency(&self, moment: f64) -> f64 {
        libm::sqrt(self.proportional / (2.0 * moment))
    }

    /// The damping ratio this gain pair gives a principal axis of moment `moment`.
    #[must_use]
    pub fn damping_ratio(&self, moment: f64) -> f64 {
        self.derivative / libm::sqrt(2.0 * moment * self.proportional)
    }
}

impl ControlLaw for QuaternionPd {
    fn torque(&self, error: &AttitudeError, _inertia: &Inertia) -> Vec3 {
        let q = error.quaternion_vector();
        let w = error.rate();
        let mut out = [0.0; 3];
        for i in 0..3 {
            out[i] = -self.proportional.mul_add(q[i], self.derivative * w[i]);
        }
        out
    }
}

impl SmallAngleGains for QuaternionPd {
    /// `q_v → δθ/2`, so the small-angle stiffness is `K/2` and not `K`. That factor is worth
    /// `√2` in the loop's delay margin, which is why it is derived here rather than at the
    /// call site.
    fn small_angle_gains(&self) -> (f64, f64) {
        (0.5 * self.proportional, self.derivative)
    }
}

/// The Schaub–Junkins MRP law, `u = −K σ − P δω + ω × Iω`.
///
/// The cross term cancels the rigid body's own gyroscopic coupling instead of overpowering it, and
/// what is left is exactly linear in the error. That is what buys the global statement: with
///
/// ```text
/// V = ½ δωᵀ I δω + 2K ln(1 + σᵀσ)
/// ```
///
/// the derivative along the closed-loop trajectory is `V̇ = −P ‖δω‖²`, negative semi-definite for
/// any positive gains and any initial error, not only small ones. [`MrpLyapunov::lyapunov`]
/// evaluates `V` so a test can assert the decrease it was measured to have rather than the decrease
/// the derivation promises.
///
/// The law is written for regulation to a fixed reference. Tracking a moving one adds the
/// feedforward `I(ω̇_ref − ω × ω_ref)`, which is additive and independent of everything here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MrpLyapunov {
    proportional: f64,
    derivative: f64,
}

impl MrpLyapunov {
    /// `None` unless both gains are finite and strictly positive.
    #[must_use]
    pub fn new(proportional: f64, derivative: f64) -> Option<Self> {
        let ok = |x: f64| x.is_finite() && x > 0.0;
        (ok(proportional) && ok(derivative)).then_some(Self {
            proportional,
            derivative,
        })
    }

    /// `V = ½ δωᵀ I δω + 2K ln(1 + σᵀσ)`, the Lyapunov function this law drives downhill.
    #[must_use]
    pub fn lyapunov(&self, error: &AttitudeError, inertia: &Inertia) -> f64 {
        let s = error.mrp();
        let w = error.rate();
        let kinetic = 0.5 * dot(w, inertia.apply(w));
        let potential = 2.0 * self.proportional * libm::log(1.0 + dot(s, s));
        kinetic + potential
    }
}

impl ControlLaw for MrpLyapunov {
    fn torque(&self, error: &AttitudeError, inertia: &Inertia) -> Vec3 {
        let s = error.mrp();
        let w = error.rate();
        let omega = error.omega();
        let gyroscopic = cross(omega, inertia.apply(omega));
        let mut out = [0.0; 3];
        for i in 0..3 {
            out[i] = (-self.proportional).mul_add(s[i], -(self.derivative * w[i])) + gyroscopic[i];
        }
        out
    }
}

impl SmallAngleGains for MrpLyapunov {
    /// `σ = q_v/(1 + q₀) → δθ/4`, half the quaternion law's stiffness at the same nominal gain. The
    /// gyroscopic term is quadratic in the rate and so contributes nothing to the linearisation
    /// about rest — the very cancellation that makes the law globally stable makes it linearise
    /// like a plain PD.
    fn small_angle_gains(&self) -> (f64, f64) {
        (0.25 * self.proportional, self.derivative)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::float_cmp)]

    use super::*;
    use spacelab_propagation::attitude::exp_so3;

    const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn inertia() -> Inertia {
        Inertia::principal([10.0, 12.0, 8.0]).expect("positive definite")
    }

    /// (G1) The quaternion convention is pinned against the crate's own verified exponential rather than
    /// asserted from a formula: `exp_so3(θ n̂)` is the rotation by `θ` about `n̂`, so its quaternion
    /// must be `(cos(θ/2), sin(θ/2) n̂)`. Getting this wrong flips the sign of every torque.
    #[test]
    fn the_quaternion_of_a_rodrigues_rotation_is_the_half_angle_pair() {
        let axis = {
            let raw = [0.3, -0.5, 0.81];
            let n = libm::sqrt(dot(raw, raw));
            [raw[0] / n, raw[1] / n, raw[2] / n]
        };
        for angle in [1e-6, 0.1, 1.0, 2.5, 3.0] {
            let m = exp_so3([axis[0] * angle, axis[1] * angle, axis[2] * angle]);
            let q = quaternion_of(&m);
            let (c, s) = (libm::cos(angle / 2.0), libm::sin(angle / 2.0));
            assert!((q[0] - c).abs() < 1e-12, "scalar part at {angle}");
            for i in 0..3 {
                assert!(
                    (q[1 + i] - s * axis[i]).abs() < 1e-12,
                    "vector part {i} at {angle}"
                );
            }
        }
    }

    /// Shepperd's branch selection exists for this case, and "exists for" is measured rather than
    /// asserted. At `θ = π − 1e-9` about `ẑ` the trace rounds to exactly `−1`, so the naive
    /// `q₀ = √(1 + tr M)/2` branch returns exactly `0.0` and every one of its three vector
    /// components is then a **division by zero**. Shepperd's rule looks at all four candidate
    /// denominators — measured here as `(0, 0, 0, 4)` — and takes the last, which needs no
    /// cancellation at all.
    ///
    /// The result is not merely survivable, it is exact: the recovered angle differs from the input
    /// by 0.000e0, and `q₀` comes back as 5.000001026e-10, which is `cos(θ/2)` to the last bit.
    #[test]
    fn the_quaternion_survives_a_rotation_of_very_nearly_pi() {
        let angle = core::f64::consts::PI - 1e-9;
        let m = exp_so3([0.0, 0.0, angle]);
        let q = quaternion_of(&m);
        let recovered = 2.0 * libm::acos(q[0].clamp(-1.0, 1.0));
        println!(
            "near-pi: |recovered - angle| = {:.3e}, q0 = {:.9e}",
            (recovered - angle).abs(),
            q[0]
        );
        assert!(
            (recovered - angle).abs() == 0.0,
            "recovered {recovered} against {angle}"
        );
        assert_eq!(q[0], libm::cos(angle / 2.0), "q0 is cos(theta/2) exactly");
        assert!((q[3] - libm::sin(angle / 2.0)).abs() < 1e-15);
    }

    /// The sign fix is not cosmetic. A rotation of 350° about `+ẑ` and one of `−10°` are the same
    /// attitude; without `q₀ ≥ 0` the proportional term points the long way and the vehicle unwinds
    /// through nearly a full revolution. The test is that the two agree, and that the error angle is
    /// the small one.
    #[test]
    fn the_error_takes_the_short_way_round() {
        let long = exp_so3([0.0, 0.0, 350.0_f64.to_radians()]);
        let short = exp_so3([0.0, 0.0, -10.0_f64.to_radians()]);
        let a = AttitudeError::new(&long, &IDENTITY, [0.0; 3], [0.0; 3]);
        let b = AttitudeError::new(&short, &IDENTITY, [0.0; 3], [0.0; 3]);
        for i in 0..4 {
            assert!((a.quaternion()[i] - b.quaternion()[i]).abs() < 1e-12);
        }
        assert!(
            (a.angle() - 10.0_f64.to_radians()).abs() < 1e-12,
            "the error angle is {} deg",
            a.angle().to_degrees()
        );
        assert!(a.quaternion()[0] >= 0.0);
        // The MRP stays inside the unit ball, which is what the sign fix buys.
        let s = a.mrp();
        assert!(libm::sqrt(dot(s, s)) <= 1.0);
    }

    #[test]
    fn an_attitude_that_is_already_the_reference_asks_for_no_torque() {
        let reference = exp_so3([0.2, -0.1, 0.4]);
        let error = AttitudeError::new(&reference, &reference, [0.0; 3], [0.0; 3]);
        assert!(error.angle() < 1e-12);
        let i = inertia();
        for t in QuaternionPd::new(1.0, 1.0)
            .expect("valid")
            .torque(&error, &i)
        {
            assert!(t.abs() < 1e-12);
        }
        for t in MrpLyapunov::new(1.0, 1.0)
            .expect("valid")
            .torque(&error, &i)
        {
            assert!(t.abs() < 1e-12);
        }
    }

    #[test]
    fn a_non_positive_gain_is_refused() {
        assert!(QuaternionPd::new(0.0, 1.0).is_none());
        assert!(QuaternionPd::new(1.0, -1.0).is_none());
        assert!(QuaternionPd::new(f64::NAN, 1.0).is_none());
        assert!(MrpLyapunov::new(1.0, 0.0).is_none());
        assert!(QuaternionPd::critically_damped(1.0, 0.0).is_none());
        assert!(QuaternionPd::critically_damped(-1.0, 1.0).is_none());
    }

    /// `critically_damped` has to invert `ζ = P/√(2IK)` and not merely produce a plausible number.
    #[test]
    fn critical_damping_round_trips_through_the_damping_ratio() {
        let (k, moment) = (0.4, 10.0);
        let law = QuaternionPd::critically_damped(k, moment).expect("valid");
        assert!((law.damping_ratio(moment) - 1.0).abs() < 1e-14);
        assert!((law.natural_frequency(moment) - libm::sqrt(k / (2.0 * moment))).abs() < 1e-15);
    }

    /// The gyroscopic term is what distinguishes the two laws, so a case where `ω × Iω` is non-zero
    /// has to separate them. A rate about an axis that is *not* principal produces the coupling; a
    /// rate about a principal axis does not, and there the two differ only through `σ` against `q_v`.
    #[test]
    fn the_lyapunov_law_carries_the_gyroscopic_term_and_the_pd_law_does_not() {
        let i = inertia();
        let attitude = exp_so3([0.05, 0.0, 0.0]);
        let spin = [0.3, 0.2, 0.1];
        let error = AttitudeError::new(&attitude, &IDENTITY, spin, [0.0; 3]);
        let expected = cross(spin, i.apply(spin));
        assert!(
            libm::sqrt(dot(expected, expected)) > 0.1,
            "the test case must actually have coupling: {expected:?}"
        );

        let pd = QuaternionPd::new(1.0, 1.0)
            .expect("valid")
            .torque(&error, &i);
        let mrp = MrpLyapunov::new(1.0, 1.0)
            .expect("valid")
            .torque(&error, &i);
        let sigma = error.mrp();
        let qv = error.quaternion_vector();
        for k in 0..3 {
            let difference = mrp[k] - pd[k];
            let predicted = expected[k] - (sigma[k] - qv[k]);
            assert!(
                (difference - predicted).abs() < 1e-12,
                "axis {k}: {difference} against {predicted}"
            );
        }
    }
    /// The three half-turns, which are the only rotations Shepperd's rule has to get right for a
    /// reason other than accuracy. Each one puts `q₀` at exactly zero, so they pin two things no
    /// other case can: that branch `k` reads `1 + 2M_kk − tr M` and not some other combination of
    /// the same three numbers, and that the `q₀ ≥ 0` convention keeps a quaternion whose scalar
    /// part *is* zero rather than flipping it to `−0` and negating the axis with it.
    ///
    /// The matrices are exact — every entry is `0` or `±1` — so the expected quaternions are exact
    /// too and the assertion is equality rather than a tolerance.
    #[test]
    fn a_half_turn_about_each_axis_selects_its_own_branch_and_keeps_the_positive_scalar() {
        let cases = [
            (
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 1.0, 0.0, 0.0],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 0.0, 1.0, 0.0],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 0.0, 1.0],
            ),
        ];
        for (m, want) in cases {
            let q = quaternion_of(&m);
            assert_eq!(q, want, "half turn {m:?}");
            assert!(
                q[0].is_sign_positive(),
                "the scalar part was negated to {:?}",
                q[0]
            );
            let error = AttitudeError::new(&m, &IDENTITY, [0.0; 3], [0.0; 3]);
            assert_eq!(error.quaternion(), want);
        }
    }

    /// The damping ratio is a function of the gain pair and the moment, not the constant `1` that
    /// [`QuaternionPd::critically_damped`] happens to produce. Two of these are deliberately not
    /// critically damped, and the reference values are the definition `ζ = P/√(2IK)` evaluated by
    /// hand: `4/√(2·2·4) = 1`, `1/√(2·2·4) = 0.25`, `8/√(2·2·4) = 2`.
    #[test]
    fn the_damping_ratio_is_the_gain_pair_measured_against_the_moment() {
        let moment = 2.0;
        for (derivative, want) in [(4.0, 1.0), (1.0, 0.25), (8.0, 2.0)] {
            let law = QuaternionPd::new(4.0, derivative).expect("valid");
            assert!(
                (law.damping_ratio(moment) - want).abs() < 1e-15,
                "zeta for D = {derivative} was {}",
                law.damping_ratio(moment)
            );
        }
    }

    /// `V = ½ δωᵀ I δω + 2K ln(1 + σᵀσ)`, both terms at once and against numbers computed outside
    /// the function. The rotation is a quarter turn about `x̂`, whose MRP is `tan(π/8) x̂`, so
    /// `σᵀσ = tan²(π/8)` exactly; the rate term is a plain diagonal quadratic form.
    #[test]
    fn the_lyapunov_function_is_the_kinetic_term_plus_the_logarithmic_potential() {
        let i = inertia();
        let attitude = exp_so3([core::f64::consts::FRAC_PI_2, 0.0, 0.0]);
        let rate = [0.3, -0.2, 0.5];
        let error = AttitudeError::new(&attitude, &IDENTITY, rate, [0.0; 3]);
        let law = MrpLyapunov::new(3.0, 1.0).expect("valid");

        let tan_eighth = libm::tan(core::f64::consts::FRAC_PI_8);
        let kinetic = 0.5 * (10.0 * 0.09 + 12.0 * 0.04 + 8.0 * 0.25);
        let potential = 2.0 * 3.0 * libm::log(1.0 + tan_eighth * tan_eighth);
        let want = kinetic + potential;
        let got = law.lyapunov(&error, &i);
        println!(
            "Lyapunov: {got:.17e} against {want:.17e} (kinetic {kinetic:.3e}, potential {potential:.3e})"
        );
        assert!((got - want).abs() < 1e-13, "V = {got:e} against {want:e}");
        assert!(
            potential > 0.1,
            "the potential term must not be negligible here"
        );
    }

    /// (G3) `σ = q_v/(1 + q₀) → δθ/4` makes the MRP law's linearised stiffness a *quarter* of its
    /// nominal gain, and the rate gain passes through untouched. A quotient in place of the product
    /// would agree at `K = ½` and nowhere else, so the fixture avoids it.
    #[test]
    fn the_mrp_law_linearises_to_a_quarter_of_its_nominal_stiffness() {
        let law = MrpLyapunov::new(8.0, 3.0).expect("valid");
        assert_eq!(law.small_angle_gains(), (2.0, 3.0));
    }

    /// `a · b` for two rotations, row by column.
    fn compose(a: &Mat3, b: &Mat3) -> Mat3 {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
            }
        }
        out
    }

    /// (G1) The error is the body's rotation relative to the reference, `[RB] = [NR]ᵀ[NB]`: a body
    /// at `[NR]·exp(b)` is `exp(b)` from its reference, so the error quaternion is
    /// `(cos(|b|/2), sin(|b|/2) b̂)` whatever the reference. `derivations/attitude_control.mac` shows
    /// this is the composition with body-rate kinematics; `[NB][NR]ᵀ`, the one used before, returns
    /// the axis `[NR] b̂` instead and passes only at an identity reference.
    #[test]
    fn the_error_is_the_body_rotation_relative_to_the_reference() {
        let reference = exp_so3([0.3, -0.7, 1.1]);
        let b = [0.4, 0.2, -0.5];
        let angle = libm::sqrt(dot(b, b));
        let actual = compose(&reference, &exp_so3(b));
        let q = AttitudeError::new(&actual, &reference, [0.0; 3], [0.0; 3]).quaternion();
        let (c, s) = (libm::cos(angle / 2.0), libm::sin(angle / 2.0));
        assert!((q[0] - c).abs() < 1e-15, "scalar part {}", q[0]);
        for k in 0..3 {
            assert!(
                (q[1 + k] - s * b[k] / angle).abs() < 1e-15,
                "vector part {k}: {} against {}",
                q[1 + k],
                s * b[k] / angle
            );
        }
    }

    /// (G2) `V̇ = −P‖ω‖²` along the true dynamics, for the shipped law, error and Lyapunov function,
    /// by a central difference of `V` along `R(h) = R·exp(hω)` — the exact flow of `Ṙ = R[ω]×` at a
    /// frozen rate — and `ω(h) = ω + h·I⁻¹(u − ω × Iω)`. `derivations/attitude_control.mac` derives
    /// the identity for any symmetric inertia and any gains, and `derivations/attitude_control.gp`
    /// re-checks it in exact rationals. The reference is a general rotation, so the composition of
    /// the error is part of what is checked.
    ///
    /// The bound is the difference's own error, `h²/6·V‴` and `ε·V/h` at `h = 1e-4`: measured
    /// 1.5e-13 against `P‖ω‖² = 3.0e-2`. With the old composition `[NB][NR]ᵀ` the same difference
    /// is off by the `Kσᵀ([NR] − I)ω` that composition leaves in `V̇`.
    #[test]
    fn the_lyapunov_derivative_along_the_body_kinematics_is_minus_the_rate_damping() {
        let moments = [10.0, 12.0, 8.0];
        let i = inertia();
        let law = MrpLyapunov::new(0.7, 1.3).expect("valid");
        let reference = exp_so3([-0.4, 0.9, 0.25]);
        let actual = compose(&reference, &exp_so3([1.1, -0.6, 0.8]));
        let omega = [0.05, -0.12, 0.08];

        let u = law.torque(
            &AttitudeError::new(&actual, &reference, omega, [0.0; 3]),
            &i,
        );
        let gyro = cross(omega, i.apply(omega));
        let omega_dot = [
            (u[0] - gyro[0]) / moments[0],
            (u[1] - gyro[1]) / moments[1],
            (u[2] - gyro[2]) / moments[2],
        ];
        let v_at = |h: f64| {
            let r = compose(
                &actual,
                &exp_so3([h * omega[0], h * omega[1], h * omega[2]]),
            );
            let w = [
                omega[0] + h * omega_dot[0],
                omega[1] + h * omega_dot[1],
                omega[2] + h * omega_dot[2],
            ];
            law.lyapunov(&AttitudeError::new(&r, &reference, w, [0.0; 3]), &i)
        };
        let h = 1e-4;
        let v_dot = (v_at(h) - v_at(-h)) / (2.0 * h);
        let want = -1.3 * dot(omega, omega);
        assert!(
            (v_dot - want).abs() < 1e-9,
            "V' = {v_dot:e} against -P|w|^2 = {want:e}"
        );
    }
}
