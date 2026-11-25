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

    /// The principal rotation angle of the error, in radians, on `[0, π]`: `2·atan2(‖q_v‖, q₀)`.
    ///
    /// Not `2·acos(q₀)`, the same function in exact arithmetic (`derivations/attitude_control.mac`
    /// item 8): `q₀ = cos(θ/2)` rounds to 1 below θ = 2e-8 rad, where that form returned 0, while
    /// `‖q_v‖ = sin(θ/2)` keeps its relative accuracy at every angle.
    /// Its round-off, under H-libm and not outward-rounded, is within 6.25e-16 of the exact angle
    /// of the stored quaternion, relative, and within 2.31e-15 of the exact angle of the error
    /// rotation [`Self::new`] formed (`derivations/rotation_angle_roundoff.mac`,
    /// `shepperd_roundoff.gappa`).
    #[must_use]
    pub fn angle(&self) -> f64 {
        let v = self.quaternion_vector();
        2.0 * libm::atan2(libm::sqrt(dot(v, v)), self.quaternion[0])
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
        // `log1p`: `1 + σᵀσ` rounds σᵀσ to an absolute u, which is all of it below 1e-16.
        let potential = 2.0 * self.proportional * libm::log1p(dot(s, s));
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
    /// The error angle's round-off through `quaternion_of`, under H-libm and not outward-rounded:
    /// within 2.31e-15 of the exact angle of the same doubles, relative, plus the fixture's rounding
    /// and the test's quotient (`derivations/rotation_angle_roundoff.mac` (4), (5), from
    /// `derivations/shepperd_roundoff.gappa`).
    const ANGLE_BOUND: f64 = 2.6e-15;

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

    /// (G3) The error angle against the rotation angle, every decade from 1e-1 to 1e-10 rad.
    ///
    /// The fixture is `derivations/attitude_control.mac` item 8: Rodrigues' `exp(θ [n]×)` for
    /// `θ = 10⁻ᵏ`, the axis cycling over three generic directions, rounded to doubles, and beside it
    /// the exact angle of those doubles at 25 digits, which rounding the matrix moved at most
    /// 5.9e-17 from `θ`, relative. `2·acos(q₀)` returned 0 for every angle up to 2e-8 rad, because
    /// `q₀ = cos(θ/2)` rounds to 1 there, and was 3.2 % off at 1e-7; `2·atan2(‖q_v‖, q₀)` reads the
    /// angle from `‖q_v‖ = sin(θ/2)`, which keeps its relative accuracy at every scale.
    #[test]
    #[allow(clippy::excessive_precision, clippy::unreadable_literal)]
    fn the_error_angle_is_the_rotation_angle_at_every_decade() {
        #[rustfmt::skip]
        let fixture: [(i32, [f64; 9], f64); 10] = [
            (1, [9.954119885206359485962935e-1, -8.618323484691077507324763e-2, -4.156228026366735189833790e-2, 8.495976511908034933817646e-2, 9.959217675738986086386717e-1, -3.035903791941082369154614e-2, 4.400921971932820336848025e-2, 2.668862873591954995577957e-2, 9.986745744615170838898166e-1], 9.999999999999999922211909e-2),
            (2, [9.999867348044213510860345e-1, 4.297887653418706530028537e-3, 2.838728044455271493085036e-3, -4.273398061581291582089470e-3, 9.999540820153048903407011e-1, -8.577408112959351849102774e-3, -2.875462432211393914993636e-3, 8.565163317040644375133240e-3, 9.999591840136042852904552e-1], 9.999999999999999853704150e-3),
            (3, [9.999995918367686753569501e-1, -2.858979115493220988890311e-4, 8.570814898010275248788070e-4, 2.855305646411588466580811e-4, 9.999998673469497889598756e-1, 4.286938061122484689390022e-4, -8.572039387704153117625294e-4, -4.284489081734729493816660e-4, 9.999995408163647736543567e-1], 1.000000000000000011880823e-3),
            (4, [9.999999954081633113389671e-1, -8.571489781632601748836281e-5, -4.285591829591938980211635e-5, 8.571367332653112504521620e-5, 9.999999959183673015061800e-1, -2.857326525850186919658000e-5, 4.285836727550918146467315e-5, 2.856959178911717831461300e-5, 9.999999986734693369072602e-1], 1.000000000000000011924442e-4),
            (5, [9.999999999867347222348712e-1, 4.285726530540815920416152e-6, 2.857124489748299327895485e-6, -4.285702040744897810525121e-6, 9.999999999540816197907134e-1, -8.571434693734693681930505e-6, -2.857161224442176916248504e-6, 8.571422448836735474017937e-6, 9.999999999591836497003783e-1], 1.000000000000000015998176e-5),
            (6, [9.999999999995918820161478e-1, -2.857144693877074615432797e-7, 8.571427959182245165858580e-7, 2.857141020407686877773318e-7, 9.999999999998673283485573e-1, 4.285715510203367253039242e-7, -8.571429183672040901946542e-7, -4.285713061223775251467726e-7, 9.999999999995408117570150e-1], 9.999999999999999961845500e-7),
            (7, [9.999999999999954480855990e-1, -8.571428632653047418668433e-8, -4.285714163265299109974323e-8, 8.571428510204068003878314e-8, 9.999999999999958921748089e-1, -2.857143040816321692609636e-8, 4.285714408163257939554560e-8, 2.857142673469383117367035e-8, 9.999999999999986677323704e-1], 1.000000000000000045410047e-7),
            (8, [1.000000000000000000000000e0, 4.285714297959183513822272e-9, 2.857142838775510224242921e-9, -4.285714273469388094085392e-9, 1.000000000000000000000000e0, -8.571428577551020049251578e-9, -2.857142875510204181028855e-9, 8.571428565306121512202525e-9, 1.000000000000000000000000e0], 9.999999999999999416880321e-9),
            (9, [1.000000000000000000000000e0, -2.857142858979592003872761e-10, 8.571428570816326846491334e-10, 2.857142855306122194603862e-10, 1.000000000000000000000000e0, 4.285714286938775326786492e-10, -8.571428572040816782914301e-10, -4.285714284489795970928442e-10, 1.000000000000000000000000e0], 1.000000000000000028963323e-9),
            (10, [1.000000000000000000000000e0, -8.571428571489795846234871e-11, -4.285714285591836551817594e-11, 8.571428571367347266182881e-11, 1.000000000000000000000000e0, -2.857142857326530486304180e-11, 4.285714285836735004391282e-11, 2.857142856959183776795928e-11, 1.000000000000000000000000e0], 1.000000000000000013351548e-10),
        ];
        for (k, e, want) in fixture {
            let r = [[e[0], e[1], e[2]], [e[3], e[4], e[5]], [e[6], e[7], e[8]]];
            let got = AttitudeError::new(&r, &IDENTITY, [0.0; 3], [0.0; 3]).angle();
            assert!(
                (got / want - 1.0).abs() <= ANGLE_BOUND,
                "θ = 1e-{k}: the error angle is {got:e} against {want:e}"
            );
        }
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

    /// (S1) The potential keeps its digits at small errors (Ф28.N, C3 row 14). The error quaternion
    /// is set to `(1, q₁, 0, 0)` so that the code's `σ = q₁/2` and `x = σᵀσ` are the doubles
    /// `derivations/log1p_expm1_sites.gp` evaluates `ln(1 + x)` on at 100 digits, for
    /// `x ≈ 10⁻ᵏ`, k = 1..18; `K = ½` and a zero rate leave `V = ln(1 + x)` with no other rounding.
    /// `ln(1 + x)` is well conditioned at 0 (`derivations/log1p_expm1_sites.mac`), so the bound is
    /// one ulp at every decade. Measured: every row rounded correctly. `ln(1 + x)` as written was
    /// 6.6 u off at 1e-1, 5.5e7 u at 1e-8, 11 % at 1e-15 and 0 — all of it — from 1e-16 down.
    #[test]
    #[allow(clippy::unreadable_literal, clippy::excessive_precision)]
    fn the_potential_keeps_its_digits_at_small_errors() {
        #[rustfmt::skip]
        const ROWS: [(f64, f64, f64); 18] = [
            (0.6324555320336758823529522, 0.1000000000000000055511151, 0.09531017980432486509042042),
            (0.2000000000000000111022302, 0.01000000000000000194289029, 0.009950330853168084771869113),
            (0.06324555320336758268418009, 0.0009999999999999998039762472, 0.0009995003330835329709814741),
            (0.02000000000000000041633363, 0.0001000000000000000047921736, 9.999500033330834012486111e-5),
            (0.006324555320336758788835052, 1.000000000000000081803054e-5, 9.999950000333331651375692e-6),
            (0.002000000000000000041633363, 9.999999999999999547481118e-7, 9.999995000003332878314906e-7),
            (0.0006324555320336758788835052, 1.000000000000000087097010e-7, 9.999999500000034204300845e-8),
            (0.0002000000000000000095843472, 1.000000000000000020922561e-8, 9.999999950000000542558937e-9),
            (6.324555320336759059885595e-5, 1.000000000000000062281591e-9, 9.999999995000000626149247e-10),
            (2.000000000000000163606108e-5, 1.000000000000000165679168e-10, 9.999999999500001656825013e-11),
            (6.324555320336758382259237e-6, 9.999999999999999394969693e-12, 9.999999999949999394970026e-12),
            (1.999999999999999909496224e-6, 9.999999999999999798866476e-13, 9.999999999994999798866480e-13),
            (6.324555320336758382259237e-7, 9.999999999999999041560007e-14, 9.999999999999499041560007e-14),
            (1.999999999999999909496224e-7, 9.999999999999998410471283e-15, 9.999999999999948410471283e-15),
            (6.324555320336758382259237e-8, 9.999999999999998804901736e-16, 9.999999999999993804901736e-16),
            (2.000000000000000041845122e-8, 1.000000000000000102357303e-16, 1.000000000000000052357303e-16),
            (6.324555320336759044003727e-9, 1.000000000000000071542424e-17, 1.000000000000000066542424e-17),
            (2.000000000000000124563183e-9, 1.000000000000000071542424e-18, 1.000000000000000071042424e-18),
        ];
        let law = MrpLyapunov::new(0.5, 1.0).expect("valid");
        let u = f64::EPSILON / 2.0;
        for (q1, x, want) in ROWS {
            let error = AttitudeError {
                quaternion: [1.0, q1, 0.0, 0.0],
                rate: [0.0; 3],
                omega: [0.0; 3],
            };
            let s = error.mrp();
            assert_eq!(dot(s, s), x, "the fixture's σᵀσ");
            let got = law.lyapunov(&error, &inertia());
            let off = (got - want).abs() / (want * u);
            assert!(
                off <= 2.0,
                "x = {x:e}: V = {got:e} against {want:e}, {off:.3} u"
            );
        }
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
