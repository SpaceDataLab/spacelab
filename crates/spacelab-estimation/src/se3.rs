//! The rigid-motion Lie group `SE(3)` and its algebra `se(3)`, for pose/extended-pose estimation.
//!
//! An `SE(3)` element is a rotation `R ∈ SO(3)` and a translation `t`, acting on a point as
//! `p ↦ R p + t`. The algebra `se(3)` is the 6-vector twist `ξ = [ρ (translation); φ (rotation)]`,
//! mapped to the group by the exponential `Exp(ξ) = (Exp(φ), V(φ) ρ)` where `V` is the SO(3) left
//! Jacobian. The rotation is carried as a 3×3 matrix so composition is ordinary matrix
//! multiplication — deliberately independent of the quaternion product-order convention used
//! elsewhere in [`crate::lie`].
//!
//! This is the group the invariant filters generalise onto: an `SE₂(3)` extended pose (rotation +
//! velocity + position, for inertial navigation) is the same machinery with a second translation
//! column, so the `Exp`/`Log`/`Adjoint` here are the reusable core. Everything is validated by the
//! group axioms themselves (associativity, inverse, `Exp`/`Log` round-trip, and the adjoint identity
//! `g Exp(ξ) g⁻¹ = Exp(Ad_g ξ)`) — group theory is its own oracle, no external reference needed.
//!
//! Gate: `derivations/se3_exponential.mac` folds the defining series `Σ Kᵏ/k!` and `Σ Kᵏ/(k+1)!`
//! through `K³ = −θ²K` onto the closed forms coded here, proves `V V⁻¹ = I`, the small-angle
//! coefficients `1/2, 1/6, 1/12`, the logarithm's trace and axis identities and `g ξ^ g⁻¹ = (Ad_g ξ)^`;
//! `derivations/se3_exponential.gp` re-sums the series at 100 digits. The module tests
//! `the_exponential_and_both_jacobians_are_their_series_definitions` (G1) and
//! `adjoint_matches_conjugation` (G1) tie the shipped maps to them.

// Index loops are clearest for the small fixed-size matrices; single-char names mirror the algebra.
#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]

use crate::lie::{M3, V3, dot, hat, mat_vec, matmul, transpose};

const EPS: f64 = 1e-8;

fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn neg3(a: V3) -> V3 {
    [-a[0], -a[1], -a[2]]
}

fn eye3() -> M3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn scale_m3(m: &M3, s: f64) -> M3 {
    let mut o = *m;
    for row in &mut o {
        for e in row {
            *e *= s;
        }
    }
    o
}

fn add_m3(a: &M3, b: &M3) -> M3 {
    let mut o = *a;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] += b[i][j];
        }
    }
    o
}

/// `Exp: so(3) → SO(3)` as a rotation matrix (Rodrigues, active convention).
#[must_use]
pub fn so3_exp(phi: V3) -> M3 {
    let theta2 = dot(phi, phi);
    let k = hat(phi);
    let k2 = matmul::<3, 3, 3>(&k, &k);
    if theta2 < EPS * EPS {
        // I + [φ]× + ½[φ]×²  (second order)
        add_m3(&add_m3(&eye3(), &k), &scale_m3(&k2, 0.5))
    } else {
        let theta = libm::sqrt(theta2);
        let a = libm::sin(theta) / theta;
        let b = (1.0 - libm::cos(theta)) / theta2;
        add_m3(&add_m3(&eye3(), &scale_m3(&k, a)), &scale_m3(&k2, b))
    }
}

/// The unit quaternion `(w, x, y, z)` of a rotation matrix, `R = (w² − v·v) I + 2 v vᵀ + 2w [v]×`,
/// sign-fixed to `w ≥ 0`.
///
/// Shepperd's branch: the four candidates `1 + tr R` and `1 + 2Rᵢᵢ − tr R` are `4w²` and `4vᵢ²`,
/// they sum to 4, and the root of the largest, at least 1, is the one divided by. The same
/// arithmetic as `spacelab_control::attitude`'s `quaternion_of`, on this module's row-major `M3`.
fn shepperd_quaternion(m: &M3) -> [f64; 4] {
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

/// `Log: SO(3) → so(3)`, the rotation vector of a rotation matrix: `2·atan2(‖v‖, w)·v/‖v‖` on
/// Shepperd's quaternion `(w, v)`, `w ≥ 0`, so the angle is in `[0, π]`.
///
/// Not `acos((tr R − 1)/2)` and `θ/(2 sin θ)·(R − Rᵀ)∨`: the first loses the angle as `u/ε²` within
/// `ε` of a half-turn, and the second is `0/0` at it, where it returned the zero vector. Here the
/// half-turn comes out as `π` to the bit, because `w = 0` there and `2·atan2(‖v‖, 0) = π`.
/// Gate: `derivations/se3_exponential.mac` item 7 (G1, G3). Round-off, under H-libm and not
/// outward-rounded: each component within 2.53e-15, relative, of the exact logarithm of the same
/// doubles (`derivations/rotation_angle_roundoff.mac`, `shepperd_roundoff.gappa`).
#[must_use]
pub fn so3_log(r: &M3) -> V3 {
    let [w, x, y, z] = shepperd_quaternion(r);
    let sin_half = libm::sqrt(dot([x, y, z], [x, y, z]));
    // `2·atan2(s, w)/s → 2/w` as `s → 0`, and `w = 1` there.
    let k = if sin_half > 0.0 {
        2.0 * libm::atan2(sin_half, w) / sin_half
    } else {
        2.0
    };
    [k * x, k * y, k * z]
}

/// The SO(3) left Jacobian `V(φ)` (relates the algebra translation to the group translation).
#[must_use]
pub fn left_jacobian(phi: V3) -> M3 {
    let theta2 = dot(phi, phi);
    let k = hat(phi);
    let k2 = matmul::<3, 3, 3>(&k, &k);
    if theta2 < EPS * EPS {
        add_m3(
            &add_m3(&eye3(), &scale_m3(&k, 0.5)),
            &scale_m3(&k2, 1.0 / 6.0),
        )
    } else {
        let theta = libm::sqrt(theta2);
        // `(1 − cos θ)/θ²` written as `2 sin²(θ/2)/θ²` — the same number, computed without the
        // cancellation. Here that matters and in `so3_exp` it does not: this coefficient scales
        // `[φ]×`, whose entries are of order θ, so losing it loses a term of order θ/2 from `V`.
        // Directly, `1 − cos θ` is below an ulp of 1 for θ near `EPS` and rounds to zero, which
        // made `V(1e-8 ẑ)` come back as the identity instead of carrying its 5e-9 off-diagonal.
        let half_sin = libm::sin(0.5 * theta);
        let a = 2.0 * half_sin * half_sin / theta2;
        let b = (theta - libm::sin(theta)) / (theta2 * theta);
        add_m3(&add_m3(&eye3(), &scale_m3(&k, a)), &scale_m3(&k2, b))
    }
}

/// The inverse left Jacobian `V(φ)⁻¹`.
#[must_use]
pub fn left_jacobian_inv(phi: V3) -> M3 {
    let theta2 = dot(phi, phi);
    let k = hat(phi);
    let k2 = matmul::<3, 3, 3>(&k, &k);
    if theta2 < EPS * EPS {
        add_m3(
            &add_m3(&eye3(), &scale_m3(&k, -0.5)),
            &scale_m3(&k2, 1.0 / 12.0),
        )
    } else {
        let theta = libm::sqrt(theta2);
        // `(1 + cos θ)/(2θ sin θ)` as `cot(θ/2)/(2θ)`: within ε of π, `1 + cos θ` is ε²/2 and
        // rounds away its digits, while `cos(θ/2)` there is computed to the ulp.
        let half = 0.5 * theta;
        let c = (1.0 / theta2) - libm::cos(half) / (2.0 * theta * libm::sin(half));
        add_m3(&add_m3(&eye3(), &scale_m3(&k, -0.5)), &scale_m3(&k2, c))
    }
}

/// A rigid motion in `SE(3)`: rotation `R` (active rotation matrix) and translation `t`.
#[derive(Clone, Copy, Debug)]
pub struct Se3 {
    rotation: M3,
    translation: V3,
}

impl Se3 {
    /// The identity motion.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            rotation: eye3(),
            translation: [0.0; 3],
        }
    }

    /// Build from a rotation matrix and translation (the matrix is trusted to be a proper rotation).
    #[must_use]
    pub fn from_parts(rotation: M3, translation: V3) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// The rotation matrix `R`.
    #[must_use]
    pub fn rotation(&self) -> M3 {
        self.rotation
    }

    /// The translation `t`.
    #[must_use]
    pub fn translation(&self) -> V3 {
        self.translation
    }

    /// Group composition `self · other`: `(R₁R₂, R₁ t₂ + t₁)`.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            rotation: matmul::<3, 3, 3>(&self.rotation, &other.rotation),
            translation: add3(mat_vec(&self.rotation, other.translation), self.translation),
        }
    }

    /// The group inverse `(Rᵀ, −Rᵀ t)`.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let rt = transpose(&self.rotation);
        Self {
            rotation: rt,
            translation: neg3(mat_vec(&rt, self.translation)),
        }
    }

    /// Act on a point: `R p + t`.
    #[must_use]
    pub fn act(&self, p: V3) -> V3 {
        add3(mat_vec(&self.rotation, p), self.translation)
    }

    /// The exponential `Exp(ξ)` of a twist `ξ = [ρ, φ]` (`ρ` = translation part, `φ` = rotation part).
    #[must_use]
    pub fn exp(rho: V3, phi: V3) -> Self {
        Self {
            rotation: so3_exp(phi),
            translation: mat_vec(&left_jacobian(phi), rho),
        }
    }

    /// The logarithm `Log(g) = (ρ, φ)` back to a twist.
    #[must_use]
    pub fn log(&self) -> (V3, V3) {
        let phi = so3_log(&self.rotation);
        let rho = mat_vec(&left_jacobian_inv(phi), self.translation);
        (rho, phi)
    }

    /// The 6×6 adjoint `Ad_g = [[R, [t]× R], [0, R]]`, mapping algebra twists `[ρ; φ]` through the
    /// conjugation `g Exp(ξ) g⁻¹ = Exp(Ad_g ξ)`.
    #[must_use]
    pub fn adjoint(&self) -> [[f64; 6]; 6] {
        let r = self.rotation;
        let tx_r = matmul::<3, 3, 3>(&hat(self.translation), &r);
        let mut ad = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                ad[i][j] = r[i][j];
                ad[i][j + 3] = tx_r[i][j];
                ad[i + 3][j + 3] = r[i][j];
            }
        }
        ad
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spacelab_testkit::nan_max;

    fn max_m3_diff(a: &M3, b: &M3) -> f64 {
        let mut m = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                m = m.max((a[i][j] - b[i][j]).abs());
            }
        }
        m
    }

    fn max_v3_diff(a: V3, b: V3) -> f64 {
        (a[0] - b[0])
            .abs()
            .max((a[1] - b[1]).abs())
            .max((a[2] - b[2]).abs())
    }

    /// `so3_log`'s round-off, under H-libm and not outward-rounded: each component within 2.53e-15
    /// of the exact logarithm of the same doubles, relative, so within 2.53e-15·π absolute, plus
    /// half an ulp for the fixture's own rounding (`derivations/rotation_angle_roundoff.mac` (3),
    /// (5), from `derivations/shepperd_roundoff.gappa`).
    const LOG_HALF_TURN_BOUND: f64 = 8.2e-15;

    fn max_se3_diff(a: &Se3, b: &Se3) -> f64 {
        max_m3_diff(&a.rotation, &b.rotation).max(max_v3_diff(a.translation, b.translation))
    }

    // A spread of twists including a large rotation and the near-identity path.
    const TWISTS: [([f64; 3], [f64; 3]); 4] = [
        ([1.0, -2.0, 0.5], [0.3, -0.4, 0.2]),
        ([-0.7, 0.1, 2.0], [1.5, 0.9, -1.2]),
        ([10.0, -5.0, 3.0], [0.01, -0.02, 0.015]),
        ([0.0, 0.0, 0.0], [2.6, -1.1, 0.4]),
    ];

    #[test]
    fn exp_log_round_trip() {
        let mut max = 0.0f64;
        for (rho, phi) in TWISTS {
            let (rho2, phi2) = Se3::exp(rho, phi).log();
            max = nan_max(nan_max(max, max_v3_diff(rho, rho2)), max_v3_diff(phi, phi2));
        }
        std::eprintln!("SE(3) Exp/Log round-trip: max err {max:.3e}");
        assert!(max < 1e-11, "Exp/Log round-trip error {max:.3e}");
    }

    #[test]
    fn composition_is_associative_and_invertible() {
        let a = Se3::exp(TWISTS[0].0, TWISTS[0].1);
        let b = Se3::exp(TWISTS[1].0, TWISTS[1].1);
        let c = Se3::exp(TWISTS[2].0, TWISTS[2].1);

        let ab_c = a.compose(&b).compose(&c);
        let a_bc = a.compose(&b.compose(&c));
        assert!(max_se3_diff(&ab_c, &a_bc) < 1e-12, "not associative");

        let id = a.compose(&a.inverse());
        assert!(max_se3_diff(&id, &Se3::identity()) < 1e-12, "inverse wrong");

        // action respects composition: (a·b)·p == a·(b·p)
        let p = [0.4, -1.3, 2.2];
        assert!(
            max_v3_diff(a.compose(&b).act(p), a.act(b.act(p))) < 1e-12,
            "action ≠ composition"
        );
    }

    /// The identity element, and the second-order series that stands in for Rodrigues below `EPS`.
    ///
    /// Every twist above turns by at least 0.4 rad, so the small-angle branch of all four maps ran
    /// on nothing — and neither did the identity, where the closed form is `0/0` in each of them:
    /// `sin θ / θ`, `θ / sin θ`, `(1 − cos θ) / θ²`, and `1/θ² − (1 + cos θ)/(2 θ sin θ)`. A round
    /// trip cannot see that: `Log` and `Exp` switch branches together and agree with each other on
    /// either. The oracle here is the closed form of a rotation about a coordinate axis, which owes
    /// nothing to Rodrigues, and `∫₀¹ Exp(sφ) ds`, which is what the left Jacobian is.
    #[test]
    #[allow(clippy::float_cmp)] // the identity element, where the maps are exact by construction
    fn the_small_angle_branch_and_the_identity_agree_with_the_closed_form() {
        assert_eq!(so3_exp([0.0; 3]), eye3(), "Exp(0)");
        assert_eq!(so3_log(&eye3()), [0.0; 3], "Log(I)");
        assert_eq!(left_jacobian([0.0; 3]), eye3(), "V(0)");
        assert_eq!(left_jacobian_inv([0.0; 3]), eye3(), "V(0)^-1");

        for theta in [
            1e-12_f64, 1e-10, 1e-9, 1e-8, 3e-8, 1e-7, 1e-6, 1e-4, 1e-2, 0.5, 1.0, 3.0,
        ] {
            let phi = [0.0, 0.0, theta];
            let (c, s) = (libm::cos(theta), libm::sin(theta));

            // Rz(θ), written out.
            let want = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
            let d = max_m3_diff(&so3_exp(phi), &want);
            assert!(d < 1e-14, "Exp about z at θ = {theta:e}: {d:.3e}");

            // Log takes it back, and the axis stays the axis.
            let back = so3_log(&so3_exp(phi));
            let d = max_v3_diff(back, phi);
            assert!(d < 1e-14, "Log∘Exp at θ = {theta:e}: {d:.3e}");

            // V(φ) = ∫₀¹ Exp(sφ) ds, which for a rotation about z integrates in closed form. The
            // off-diagonal is written as `2 sin²(θ/2)/θ` rather than `(1 − cos θ)/θ`: the latter is
            // the same number and loses every digit of it below θ ≈ 1e-8, where `cos θ` rounds to 1.
            let half = libm::sin(0.5 * theta);
            let (sinc, versinc) = (s / theta, 2.0 * half * half / theta);
            let want = [[sinc, -versinc, 0.0], [versinc, sinc, 0.0], [0.0, 0.0, 1.0]];
            let d = max_m3_diff(&left_jacobian(phi), &want);
            assert!(d < 1e-14, "V about z at θ = {theta:e}: {d:.3e}");

            // And the inverse is the inverse.
            let d = max_m3_diff(
                &matmul::<3, 3, 3>(&left_jacobian(phi), &left_jacobian_inv(phi)),
                &eye3(),
            );
            assert!(d < 1e-13, "V·V⁻¹ at θ = {theta:e}: {d:.3e}");
        }
    }

    /// (G3) The logarithm at and near the half-turn, against the rotation vector it inverts.
    ///
    /// `acos` of the trace loses the angle as `u/ε²` within `ε` of `π`, and `θ/(2 sin θ)` is `0/0`
    /// at `π`: the half-turn `diag(1, −1, −1)` came back as the zero vector, and the error was 8.5e-9
    /// at `ε = 1e-4` and 1.2e-2 at `ε = 1e-7`. The fixture is `derivations/se3_exponential.mac`
    /// item 7: `Exp(θ n)` for `θ = π − 10⁻ᵏ`, k = 1..10, rounded to doubles, the axis cycling so
    /// that each of Shepperd's three vector branches is taken, and beside it the exact logarithm of
    /// those doubles at 25 digits, which rounding the matrix moved at most 6.1e-17 from `θ n`. The
    /// tolerance is then the proved round-off alone.
    #[test]
    #[allow(
        clippy::excessive_precision,
        clippy::unreadable_literal,
        clippy::float_cmp
    )]
    fn the_logarithm_near_and_at_the_half_turn_is_the_rotation_vector() {
        // Exactly a half-turn about each coordinate axis: π to the bit, as `2·atan2(1, 0)` is.
        for (axis, r) in [
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        ]
        .iter()
        .enumerate()
        {
            let mut want = [0.0; 3];
            want[axis] = core::f64::consts::PI;
            assert_eq!(so3_log(r), want, "half-turn about axis {axis}");
        }
        // About (x̂ + ŷ)/√2: two of Shepperd's candidates tie at 2.
        let phi = so3_log(&[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]);
        let norm = libm::sqrt(dot(phi, phi));
        assert!(
            (norm - core::f64::consts::PI).abs() <= 2.0 * f64::EPSILON * core::f64::consts::PI,
            "|Log| = {norm} at the half-turn about (1, 1, 0)/√2"
        );
        assert!(
            (phi[0] - phi[1]).abs() <= 2.0 * f64::EPSILON && phi[2] == 0.0,
            "{phi:?}"
        );

        #[rustfmt::skip]
        let fixture: [(i32, [f64; 9], [f64; 3]); 10] = [
            (1, [-8.321466823981868987658572e-1, -3.298577243027538408171040e-1, 4.457866986480187310171175e-1, -1.587147243367627025278921e-1, -6.285748287983883297869170e-1, -7.613825062869399307174945e-1, 5.313581986310143001617234e-1, -7.043348396316095882951913e-1, 4.707131806405237539614461e-1], [8.690264724542266626139757e-1, -1.303539708681339969236467e0, 2.607079417362679889103940e0]),
            (2, [4.694010202976194312540770e-1, 4.940693164289282068857290e-1, -7.318184151188089625605926e-1, 4.854980307139282191819518e-1, -8.366887758928558405102649e-1, -2.534631225007141219762730e-1, -7.375326055954757320520798e-1, -2.363205510707141188131430e-1, -6.326122452380941174610030e-1], [2.684222274505537066513231e0, 8.947407581685123217016710e-1, -1.342111137252768533256616e0]),
            (3, [-6.326526530612585075274978e-1, -7.349794081156615366268170e-1, -2.440407552449030403884933e-1, -7.344079796394711312501613e-1, 4.693878877550909933802359e-1, 4.902243672755204340063528e-1, -2.457550406734745063186409e-1, 4.893672245612347149190668e-1, -8.367342346939158348462229e-1], [-1.345968280109911377999050e0, 2.691936560219822755998099e0, 8.973121867399409525650804e-1]),
            (4, [-8.367346892857142615085309e-1, -2.449836728570000121685268e-1, 4.897530600000714051667217e-1, -2.448122442858571556900671e-1, -6.326530571428571336767277e-1, -7.347224471428095426972504e-1, 4.898387742856428750393150e-1, -7.346653042857619331229557e-1, 4.693877564285714454328513e-1], [8.975693295970837763004945e-1, -1.346353994395625715313872e0, 2.692707988791251328901484e0]),
            (5, [4.693877551153061156163915e-1, 4.898002040693876790733441e-1, -7.346910203897959457108868e-1, 4.897916326408163767425208e-1, -8.367346938316326809825796e-1, -2.449065306061223024070017e-1, -7.346967346755101102573349e-1, -2.448893877489797255009307e-1, -6.326530611836734818709260e-1], [2.692785131648394237989632e0, 8.975950438827980454201526e-1, -1.346392565824197042697935e0]),
            (6, [-6.326530612240816031643931e-1, -7.346941632651224907135656e-1, -2.448971020407550991393464e-1, -7.346935918365510298499998e-1, 4.693877551021734539915542e-1, 4.897963469386530377391864e-1, -2.448988163264693984633169e-1, 4.897954897957959019549889e-1, -8.367346938770918729488812e-1], [-1.346396422967054287256023e0, 2.692792845934108472782581e0, 8.975976153113694570177051e-1]),
            (7, [-8.367346938775463982551628e-1, -2.448980448979585777813384e-1, 4.897958755102028716166274e-1, -2.448978734693871561756140e-1, -6.326530612244857243453566e-1, -7.346939061224471156918980e-1, 4.897959612244885962972774e-1, -7.346938489795900029122322e-1, 4.693877551020421701188923e-1], [8.975978724542266749775086e-1, -1.346396808681339936169142e0, 2.692793617362679923203031e0]),
            (8, [4.693877551020408378512627e-1, 4.897959226530612109051788e-1, -7.346938746938774800199212e-1, 4.897959140816326439882289e-1, -8.367346938775509501695637e-1, -2.448979677551020306403018e-1, -7.346938804081633023201903e-1, -2.448979506122448968064020e-1, -6.326530612244897211482453e-1], [2.692793694505537052556301e0, 8.975978981685123678070165e-1, -1.346396847252768551710525e0]),
            (9, [-6.326530612244898321705477e-1, -7.346938778367346767339541e-1, -2.448979583265306070316569e-1, -7.346938772653061056061574e-1, 4.693877551020408378512627e-1, 4.897959187959183835481269e-1, -2.448979600408163204150469e-1, 4.897959179387755268564320e-1, -8.367346938775510611918662e-1], [-1.346396851109911325854777e0, 2.692793702219822804303800e0, 8.975979007399409347679333e-1]),
            (10, [-8.367346938775510611918662e-1, -2.448979592693877604947517e-1, 4.897959183244898206943674e-1, -2.448979590979591947075278e-1, -6.326530612244898321705477e-1, -7.346938775795918141753305e-1, 4.897959184102040897101915e-1, -7.346938775224489681647810e-1, 4.693877551020408378512627e-1], [8.975979009970837927006276e-1, -1.346396851495625612753818e0, 2.692793702991251378101883e0]),
        ];
        for (k, e, want) in fixture {
            let r = [[e[0], e[1], e[2]], [e[3], e[4], e[5]], [e[6], e[7], e[8]]];
            let got = so3_log(&r);
            let d = max_v3_diff(got, want);
            assert!(
                d <= LOG_HALF_TURN_BOUND,
                "θ = π − 1e-{k}: Log off by {d:.3e} ({got:?} against {want:?})"
            );
        }
    }

    /// A pose reports the motion it was built from, and acting is that motion.
    ///
    /// `rotation`, `translation` and `act` are read by every other test only through quantities
    /// they also produce, so all three survive being replaced by zero: a composition of zeros is
    /// still associative, and a zero pose is still its own inverse. Here they are read against the
    /// parts that went in.
    #[test]
    #[allow(clippy::float_cmp)] // the accessors return their input, and returning it is the claim
    fn a_pose_reports_the_parts_it_was_built_from_and_acts_with_them() {
        let r = so3_exp([0.3, -0.2, 0.7]);
        let t = [4.0, -5.0, 6.0];
        let g = Se3::from_parts(r, t);
        assert_eq!(g.rotation(), r);
        assert_eq!(g.translation(), t);

        // The origin goes to the translation, and each basis vector picks out a column of R.
        assert_eq!(g.act([0.0; 3]), t);
        for j in 0..3 {
            let mut e = [0.0; 3];
            e[j] = 1.0;
            let column = [r[0][j], r[1][j], r[2][j]];
            let moved = g.act(e);
            let d = max_v3_diff([moved[0] - t[0], moved[1] - t[1], moved[2] - t[2]], column);
            assert!(d < 1e-15, "column {j}: {d:.3e}");
        }

        // The inverse motion undoes it.
        for p in [[0.0; 3], [1.0, 2.0, 3.0], [-7.5, 0.25, 11.0]] {
            let d = max_v3_diff(g.inverse().act(g.act(p)), p);
            assert!(d < 1e-14, "round trip through the action: {d:.3e}");
        }
    }

    /// The defining adjoint identity `g Exp(ξ) g⁻¹ = Exp(Ad_g ξ)` (G1), proved on the algebra in
    /// `derivations/se3_exponential.mac`.
    #[test]
    fn adjoint_matches_conjugation() {
        let g = Se3::exp(TWISTS[1].0, TWISTS[1].1);
        let (rho, phi) = (TWISTS[0].0, TWISTS[0].1);
        let xi = [rho[0], rho[1], rho[2], phi[0], phi[1], phi[2]];

        // left: g · Exp(ξ) · g⁻¹
        let conj = g.compose(&Se3::exp(rho, phi)).compose(&g.inverse());

        // right: Exp(Ad_g ξ)
        let ad = g.adjoint();
        let mut adxi = [0.0; 6];
        for i in 0..6 {
            for j in 0..6 {
                adxi[i] += ad[i][j] * xi[j];
            }
        }
        let rhs = Se3::exp([adxi[0], adxi[1], adxi[2]], [adxi[3], adxi[4], adxi[5]]);

        let d = max_se3_diff(&conj, &rhs);
        std::eprintln!("SE(3) adjoint vs conjugation: max diff {d:.3e}");
        assert!(d < 1e-11, "Ad_g does not match conjugation ({d:.3e})");
    }

    /// The left Jacobian and its inverse, against the closed forms at 60 digits.
    ///
    /// Below `‖φ‖ = 1e-8` both are second-order series, and their quadratic coefficients — `1/6`
    /// and `1/12` — move a matrix of order one by about `1e-17`, which every tolerance in this
    /// module is eight decades too coarse to see. On an *off-diagonal* entry they sit under a
    /// first-order term of order `θ/2`, and a relative comparison there resolves them with four
    /// decades to spare. That is the branch which has already been wrong once here, so it is the
    /// one that wants an external reference rather than a self-consistency check.
    ///
    /// Reference: `tools/gen_se3_oracle.mac`, exact rational input, bigfloat output, evaluating
    /// `V = I + ((1−cos θ)/θ²)[φ]× + ((θ−sin θ)/θ³)[φ]×²` and
    /// `V⁻¹ = I − ½[φ]× + (1/θ² − (1+cos θ)/(2θ sin θ))[φ]×²` — no series, no shared arithmetic.
    #[test]
    // The four tables are machine-generated at full f64 precision; grouping their digits would
    // mean hand-editing 36 numbers whose whole point is that nobody edits them.
    #[allow(clippy::unreadable_literal)]
    fn the_left_jacobian_matches_the_closed_form_on_both_sides_of_the_small_angle_split() {
        // Negative powers of two, so the f64 passed in is the rational the oracle evaluated.
        const SMALL: V3 = [
            1.0 / 268_435_456.0,
            1.0 / 536_870_912.0,
            1.0 / 1_073_741_824.0,
        ];
        const LARGE: V3 = [0.3, -0.7, 1.1];
        const V_SMALL: [[f64; 3]; 3] = [
            [1.0, -4.6566128615125696e-10, 9.313225751937197e-10],
            [4.656612884642216e-10, 1.0, -1.8626451489418365e-09],
            [-9.313225740372374e-10, 1.8626451495200777e-09, 1.0],
        ];
        const V_INV_SMALL: [[f64; 3]; 3] = [
            [1.0, 4.656612878859804e-10, -9.313225743263579e-10],
            [-4.656612867294981e-10, 1.0, 1.862645149375517e-09],
            [9.31322574904599e-10, -1.8626451490863967e-09, 1.0],
        ];
        const V_LARGE: [[f64; 3]; 3] = [
            [0.740970688329687, -0.5046978247763909, -0.2505269853112544],
            [
                0.44070234777549017,
                0.8019187616638784,
                -0.24624324651630197,
            ],
            [0.3510913063126699, 0.011593164179665608, 0.9116252936654226],
        ];
        const V_INV_LARGE: [[f64; 3]; 3] = [
            [0.8539183775499095, 0.5319546231091064, 0.37835702082854694],
            [-0.5680453768908935, 0.8882905240087543, 0.08383361806672374],
            [
                -0.32164297917145296,
                -0.21616638193327625,
                0.950160387634675,
            ],
        ];
        fn check(got: &M3, want: &[[f64; 3]; 3], rel: f64, what: &str) {
            for i in 0..3 {
                for j in 0..3 {
                    let (g, w) = (got[i][j], want[i][j]);
                    assert!(
                        (g - w).abs() <= rel * w.abs(),
                        "{what}[{i}][{j}] = {g:.17e}, oracle says {w:.17e}",
                    );
                }
            }
        }

        // θ = 4.27e-9: the series branch, where the mutation signal is 1e-8 relative and the
        // truncation-plus-rounding floor is 3e-16.
        check(&left_jacobian(SMALL), &V_SMALL, 1e-12, "V small");
        check(
            &left_jacobian_inv(SMALL),
            &V_INV_SMALL,
            1e-12,
            "V-inv small",
        );
        // θ = 1.338: the closed-form branch, at f64 rounding.
        check(&left_jacobian(LARGE), &V_LARGE, 1e-14, "V large");
        check(
            &left_jacobian_inv(LARGE),
            &V_INV_LARGE,
            1e-14,
            "V-inv large",
        );
    }

    /// (S1) `V⁻¹` about the half-turn (Ф28.N, C3 row 13). `φ = θ·(2, −3, 6)/7` rounded to
    /// doubles for `θ = π ∓ 10⁻ᵏ`, k = 1..8, and `V⁻¹` of those doubles at 100 digits, which
    /// `derivations/se3_exponential.gp` also sums from the Bernoulli series. Within `ε` of `π`,
    /// `1 + cos θ` in `1/θ² − (1 + cos θ)/(2θ sin θ)` is `ε²/2` and keeps only `u/(ε²/2)` of its
    /// digits; written `cos(θ/2)/(2θ sin(θ/2))`, the same `cot(θ/2)/(2θ)`, nothing cancels.
    /// Measured: every entry within 2.75 u (u = 2⁻⁵³); the `1 + cos θ` form was 5.3 u and 7.0 u at
    /// `10⁻¹`, 3.4e3 u at `10⁻⁴` and 6.5e7 u, 7.2e-9, at `10⁻⁸`, on both sides.
    #[test]
    #[allow(clippy::unreadable_literal, clippy::excessive_precision)]
    fn the_inverse_jacobian_keeps_its_digits_about_the_half_turn() {
        #[rustfmt::skip]
        const ROWS: [(f64, V3, M3); 16] = [
        (-1e-1, [0.8690264724542265950546494, -1.303539708681340059115428, 2.607079417362680118230855], [[0.1515233894445454817918686, 1.190409493940612804473938, 0.8780302838221245388406932], [-1.416669923422067313756917, 0.2457985683951515586411203, 0.09512259200493149024407210], [-0.4255094248592155202747345, -0.7739038804492951048105773, 0.7548845347284242695659992]]),
        (-1e-2, [0.8947407581685123956205530, -1.342111137252768537919678, 2.684222274505537075839356], [[0.08882259406992495514779129, 1.220620816462091860247748, 0.9140362102077376243037005], [-1.463601458043445215591609, 0.1900645280621555089869662, 0.08289941671222617986939301], [-0.4280749270450309136159777, -0.8118413414562862157511600, 0.7367709716202005358982914]]),
        (-1e-3, [0.8973121867399409534726828, -1.345968280109911319186722, 2.691936560219822638373444], [[0.08235370755708564497367063, 1.223615441117522728424261, 0.9176898180397328411182830], [-1.468321119102299909949183, 0.1843144067174094487424399, 0.08159757639280473472580968], [-0.4282784620701784780684388, -0.8157146103471362187468732, 0.7349021821831580617582375]]),
        (-1e-4, [0.8975693295970837981556656, -1.346353994395625752744650, 2.692707988791251505489299], [[0.08170477916812708598698645, 1.223914631618042702591178, 0.9180557227529789766792677], [-1.468793357173208802898121, 0.1837375814827796387194507, 0.08146657646579273347262737], [-0.4282982716426467760653819, -0.8161027531312910646830382, 0.7347147139819033871272588]]),
        (-1e-5, [0.8975950438827980937261941, -1.346392565824197085078140, 2.692785131648394170156280], [[0.08163986587854149261260718, 1.223944547941335945711345, 0.9180923186778208212726599], [-1.468840583707058224444935, 0.1836798807809257644799052, 0.08145346829281564390813993], [-0.4283002471463762638054801, -0.8161415755899824498180542, 0.7346959612538008689123409]]),
        (-1e-6, [0.8975976153113695010787865, -1.346396422967054196107028, 2.692792845934108392214057], [[0.08163337434502258885599844, 1.223947539546390536239331, 0.9180959783248544177889089], [-1.468845306387717855974726, 0.1836741105289089611406691, 0.08145215739369378608179261], [-0.4283004446421997783181195, -0.8161454579176757149969939, 0.7346940859218954078270700]]),
        (-1e-7, [0.8975978724542266196095852, -1.346396808681340040436680, 2.692793617362680080873361], [[0.08163272518962482127368018, 1.223947838706623360036836, 0.9180963442901033810180284], [-1.468845778656056720836524, 0.1836735335018887434837137, 0.08145202630296323831426468], [-0.4283004643912366594186519, -0.8161458461512633812953205, 0.7346938983881138507195056]]),
        (-1e-8, [0.8975978981685123203604348, -1.346396847252768536051803, 2.692793694505537072103607], [[0.08163266027406473549069624, 1.223947868622643839165729, 0.9180963808866336617980505], [-1.468845825882893232937878, 0.1836734757991686605008404, 0.08145201319388205437649592], [-0.4283004663661348742537530, -0.8161458849746302659839389, 0.7346938796347298192064227]]),
        (1e-1, [0.9261693295970837569441869, -1.389253994395625690927432, 2.778507988791251381854863], [[0.007146249582190568636049573, 1.256873494339917772035169, 0.9593879973092286832482416], [-1.521634494451333609819695, 0.1174633329619471791737291, 0.06594316463141810592651329], [-0.4298659970863970076791900, -0.8602261649656656510176736, 0.7131755832126328379920994]]),
        (1e-2, [0.9004550438827980674005858, -1.350682565824197212123181, 2.701365131648394424246362], [[0.07439679330252000123590709, 1.227268804931199889098899, 0.9221688046980932521101549], [-1.474096326717194535147463, 0.1772415940466844590687132, 0.07998623926240703419465571], [-0.4285137611261039600130262, -0.8204688046203910332059300, 0.7326035180651724583271690]]),
        (1e-3, [0.8978836153113695095484559, -1.346825422967054208811533, 2.693650845934108417623065], [[0.08091113938165177221720763, 1.224280241551274440056302, 0.9185030743150866419162271], [-1.469370604382833977566763, 0.1830321238948015685697071, 0.08130626340834546366111099], [-0.4283223486519675668953055, -0.8165773519030240458873449, 0.7344854402658105052393825]]),
        (1e-4, [0.8976264724542266648654731, -1.346439708681339997298210, 2.692879417362679994596419], [[0.08156052236248495949679237, 1.223981111663004658564449, 0.9181370483773406761166269], [-1.468898305699675336031971, 0.1836093532110977417749265, 0.08143744517210731623145352], [-0.4283026603039993211815828, -0.8161890272821193486340196, 0.7346730397936067660768511]]),
        (1e-5, [0.8976007581685123692949446, -1.346401137252768442920114, 2.692802274505536885840229], [[0.08162544019798929196218590, 1.223951195945833671751360, 0.9181004512402537637975666], [-1.468851078559703214088869, 0.1836670579537682460592148, 0.08145055516345190143234667], [-0.4283006860125146791225478, -0.8161502030050604678625980, 0.7346917938349746708819033]]),
        (1e-6, [0.8975981867399409619423523, -1.346397280109911331891226, 2.692794560219822663782452], [[0.08163193177696737939191411, 1.223948204346840305713144, 0.9180967915810977183017766], [-1.468846355872982358069308, 0.1836728282461932126634743, 0.08145186608075743272794178], [-0.4283004885288136135894493, -0.8161463206591835292144105, 0.7346936691800127850283258]]),
        (1e-7, [0.8975979295970837323892511, -1.346396894395625709606179, 2.692793788791251419212358], [[0.08163258093281937522365962, 1.223947905186668303066333, 0.9180964256157276678827814], [-1.468845883604583116146025, 0.1836734052736172358836967, 0.08145199717166961628408937], [-0.4283004687798980417233977, -0.8161459324254141161051617, 0.7346938567139256107495009]]),
        (1e-8, [0.8975979038827980316384014, -1.346396855824196991946451, 2.692793711648393983892902], [[0.08163264584838437484640949, 1.223947875270648236877473, 0.9180963890191960061111820], [-1.468845836377745747015429, 0.1836734629763416597988091, 0.08145201028075276575776451], [-0.4283004668050009858352689, -0.8161458936020452658806369, 0.7346938754673110348909634]]),
        ];
        let u = f64::EPSILON / 2.0;
        for (offset, phi, want) in ROWS {
            let error = max_m3_diff(&left_jacobian_inv(phi), &want) / u;
            assert!(error <= 4.0, "θ − π = {offset:e}: V⁻¹ is {error:.3} u off");
        }
    }

    /// `Exp`, `V`, `V⁻¹` and the SE(3) translation `V ρ` against their defining series (G1).
    ///
    /// Reference: `derivations/se3_exponential.mac` sums `Σ Kᵏ/k!`, `Σ Kᵏ/(k+1)!` and the Bernoulli
    /// series `Σ Bₖ/k! Kᵏ` to `k = 80` at 50 digits for `φ = (3/10, −7/10, 11/10)`, `ρ = (1, −2, 1/2)`
    /// — no closed form enters — and `derivations/se3_exponential.gp` agrees with the closed forms
    /// there to 1e-90. The literals are those sums rounded once to f64.
    #[test]
    // Machine-generated at full f64 precision from the derivation; grouping the digits would mean
    // hand-editing numbers whose whole point is that nobody edits them.
    #[allow(clippy::unreadable_literal)]
    fn the_exponential_and_both_jacobians_are_their_series_definitions() {
        const PHI: V3 = [0.3, -0.7, 1.1];
        const RHO: V3 = [1.0, -2.0, 0.5];
        const EXP: M3 = [
            [
                0.26946350302809186,
                -0.8902258527560322,
                -0.3672701343978637,
            ],
            [0.7097403652688549, 0.44135444349207026, -0.5490672719420067],
            [0.6508901861634281, -0.11271284884431015, 0.7507581363272313],
        ];
        const V: M3 = [
            [0.7409706883296872, -0.5046978247763909, -0.2505269853112544],
            [
                0.44070234777549017,
                0.8019187616638784,
                -0.24624324651630197,
            ],
            [
                0.35109130631266994,
                0.011593164179665607,
                0.9116252936654227,
            ],
        ];
        const V_INV: M3 = [
            [0.8539183775499095, 0.5319546231091065, 0.37835702082854694],
            [-0.5680453768908935, 0.8882905240087544, 0.08383361806672374],
            [-0.321642979171453, -0.21616638193327625, 0.950160387634675],
        ];
        const TRANSLATION: V3 = [1.625102845226842, -1.2862567988104177, 0.7837176247860501];

        let d = max_m3_diff(&so3_exp(PHI), &EXP);
        assert!(d < 1e-15, "Exp against its series: {d:.3e}");
        let d = max_m3_diff(&left_jacobian(PHI), &V);
        assert!(d < 1e-15, "V against its series: {d:.3e}");
        let d = max_m3_diff(&left_jacobian_inv(PHI), &V_INV);
        assert!(d < 1e-15, "V⁻¹ against its Bernoulli series: {d:.3e}");
        let g = Se3::exp(RHO, PHI);
        let d = max_m3_diff(&g.rotation(), &EXP).max(max_v3_diff(g.translation(), TRANSLATION));
        assert!(d < 4e-15, "SE(3) Exp against its series: {d:.3e}");
    }
}
