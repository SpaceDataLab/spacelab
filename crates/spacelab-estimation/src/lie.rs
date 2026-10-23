//! Minimal Lie-group primitives for on-manifold estimation: unit quaternions and the group `SO(3)`.
//!
//! Array-based (`[f64; 3]`, `[f64; 4]`) to match the estimation crate's idiom (its filters carry
//! `[f64; N]` state, not `Vec3`), so this module adds no cross-crate dependency. The quaternion is
//! Hamilton convention `q = w + xi + yj + zk` with the **attitude (passive)** rotation matrix
//! `A(q)` — the map that resolves an inertial vector in the body frame, `v_body = A(q) v_inertial`.
//! For a small rotation vector `δθ`, `A(exp(δθ)) ≈ I − [δθ]×`, which is the sign the multiplicative
//! and invariant filters here assume (verified end-to-end by the Wahba/QUEST convergence test).

// Single-char names (w, x, y, z, a, b) mirror the quaternion algebra; index loops are clearest for
// the small fixed-size matrix helpers.
#![allow(clippy::many_single_char_names, clippy::needless_range_loop)]

/// A 3-vector.
pub(crate) type V3 = [f64; 3];
/// A 3×3 matrix, row-major.
pub(crate) type M3 = [[f64; 3]; 3];

#[inline]
pub(crate) fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub(crate) fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub(crate) fn norm(a: V3) -> f64 {
    libm::sqrt(dot(a, a))
}

/// The skew-symmetric matrix `[w]×` with `[w]× v = w × v`.
#[inline]
pub(crate) fn hat(w: V3) -> M3 {
    [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
}

#[inline]
pub(crate) fn mat_vec(m: &M3, v: V3) -> V3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// --- small fixed-size dense linear algebra, shared by the on-manifold filters ---

/// Generic dense matrix product `A(P×Q) · B(Q×R)`.
pub(crate) fn matmul<const P: usize, const Q: usize, const R: usize>(
    a: &[[f64; Q]; P],
    b: &[[f64; R]; Q],
) -> [[f64; R]; P] {
    let mut out = [[0.0; R]; P];
    for i in 0..P {
        for k in 0..Q {
            let aik = a[i][k];
            for j in 0..R {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

/// Transpose an `R×C` matrix.
pub(crate) fn transpose<const R: usize, const C: usize>(m: &[[f64; C]; R]) -> [[f64; R]; C] {
    let mut out = [[0.0; R]; C];
    for i in 0..R {
        for j in 0..C {
            out[j][i] = m[i][j];
        }
    }
    out
}

pub(crate) fn mat_add<const R: usize, const C: usize>(
    a: &[[f64; C]; R],
    b: &[[f64; C]; R],
) -> [[f64; C]; R] {
    let mut out = *a;
    for i in 0..R {
        for j in 0..C {
            out[i][j] += b[i][j];
        }
    }
    out
}

pub(crate) fn mat_sub<const R: usize, const C: usize>(
    a: &[[f64; C]; R],
    b: &[[f64; C]; R],
) -> [[f64; C]; R] {
    let mut out = *a;
    for i in 0..R {
        for j in 0..C {
            out[i][j] -= b[i][j];
        }
    }
    out
}

pub(crate) fn eye<const N: usize>() -> [[f64; N]; N] {
    let mut m = [[0.0; N]; N];
    for i in 0..N {
        m[i][i] = 1.0;
    }
    m
}

/// Inverse of a 3×3 matrix (cofactor expansion); `None` if singular.
pub(crate) fn inv3(m: &M3) -> Option<M3> {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];
    let ca = e * i - f * h;
    let cb = -(d * i - f * g);
    let cc = d * h - e * g;
    let det = a * ca + b * cb + c * cc;
    if libm::fabs(det) < 1e-300 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [ca * inv, -(b * i - c * h) * inv, (b * f - c * e) * inv],
        [cb * inv, (a * i - c * g) * inv, -(a * f - c * d) * inv],
        [cc * inv, -(a * h - b * g) * inv, (a * e - b * d) * inv],
    ])
}

/// A unit quaternion (Hamilton, scalar-first `w + xi + yj + zk`) representing an attitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitQuaternion {
    /// Scalar part.
    pub w: f64,
    /// `i` component.
    pub x: f64,
    /// `j` component.
    pub y: f64,
    /// `k` component.
    pub z: f64,
}

impl UnitQuaternion {
    /// The identity rotation.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// From raw scalar-first components (normalized on construction).
    #[must_use]
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }.normalized()
    }

    /// The vector (imaginary) part.
    #[must_use]
    fn vector(self) -> V3 {
        [self.x, self.y, self.z]
    }

    /// Renormalize to unit length (guards against drift after many products).
    #[must_use]
    pub fn normalized(self) -> Self {
        let n = libm::sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z);
        let inv = if n > 0.0 { 1.0 / n } else { 1.0 };
        Self {
            w: self.w * inv,
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }

    /// The conjugate (inverse, for a unit quaternion).
    #[must_use]
    pub const fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Hamilton product `self ⊗ other` — the `SO(3)` group composition.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, o: Self) -> Self {
        Self {
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        }
    }

    /// The attitude (passive) rotation matrix `A(q)`: `v_body = A(q) v_inertial`.
    #[must_use]
    pub fn attitude_matrix(self) -> M3 {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let v = [x, y, z];
        let vv = dot(v, v);
        let ww = w * w - vv;
        // A = (w² − |v|²) I + 2 v vᵀ − 2w [v]×  (passive / inertial→body)
        let outer = |i: V3, a: usize, b: usize| 2.0 * i[a] * i[b];
        let sk = hat(v);
        [
            [
                ww + outer(v, 0, 0),
                outer(v, 0, 1) - 2.0 * w * sk[0][1],
                outer(v, 0, 2) - 2.0 * w * sk[0][2],
            ],
            [
                outer(v, 1, 0) - 2.0 * w * sk[1][0],
                ww + outer(v, 1, 1),
                outer(v, 1, 2) - 2.0 * w * sk[1][2],
            ],
            [
                outer(v, 2, 0) - 2.0 * w * sk[2][0],
                outer(v, 2, 1) - 2.0 * w * sk[2][1],
                ww + outer(v, 2, 2),
            ],
        ]
    }

    /// Resolve the inertial vector `v` in the body frame: `A(q) v`.
    #[must_use]
    pub fn rotate(self, v: V3) -> V3 {
        mat_vec(&self.attitude_matrix(), v)
    }

    /// `exp: so(3) → SO(3)` as a unit quaternion, from a rotation vector `φ` (axis·angle, rad).
    #[must_use]
    pub fn from_rotation_vector(phi: V3) -> Self {
        let angle = norm(phi);
        if angle < 1e-8 {
            // second-order small-angle: q ≈ [1 − |φ|²/8, φ/2]
            Self {
                w: 1.0 - angle * angle / 8.0,
                x: 0.5 * phi[0],
                y: 0.5 * phi[1],
                z: 0.5 * phi[2],
            }
            .normalized()
        } else {
            let half = 0.5 * angle;
            let s = libm::sin(half) / angle;
            Self {
                w: libm::cos(half),
                x: phi[0] * s,
                y: phi[1] * s,
                z: phi[2] * s,
            }
        }
    }

    /// `log: SO(3) → so(3)`, the rotation vector (axis·angle, rad) of this quaternion.
    #[must_use]
    pub fn to_rotation_vector(self) -> V3 {
        let q = if self.w < 0.0 {
            // pick the short-way representative (q and −q are the same rotation)
            Self {
                w: -self.w,
                x: -self.x,
                y: -self.y,
                z: -self.z,
            }
        } else {
            self
        };
        let v = q.vector();
        let sin_half = norm(v);
        if sin_half < 1e-8 {
            scale(v, 2.0)
        } else {
            let angle = 2.0 * libm::atan2(sin_half, q.w);
            scale(v, angle / sin_half)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use spacelab_testkit::assert_within_ulps;

    /// A unit axis with three distinct non-zero components, so a mutation of any one shows.
    const AXIS: V3 = [2.0 / 7.0, -3.0 / 7.0, 6.0 / 7.0];

    #[test]
    fn the_inverse_refuses_a_singular_matrix_and_accepts_the_determinant_at_its_threshold() {
        assert!(
            inv3(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).is_none(),
            "the third row is the second plus (3,3,3) minus the first: the determinant is zero"
        );
        // `1e-300` is the guard's own constant, reached exactly: the cofactor expansion is
        // `a·1 + b·(−0) + c·0`, so the determinant is the literal itself. The guard admits it.
        let smallest = inv3(&[[1e-300, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            .expect("a determinant of exactly 1e-300 is not singular");
        assert!(smallest[0][0] > 9e299, "got {}", smallest[0][0]);

        let m = [[2.0, -1.0, 3.0], [0.5, 4.0, -2.0], [-1.5, 0.25, 1.0]];
        let inv = inv3(&m).expect("well conditioned");
        let product = matmul(&inv, &m);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i][j] - want).abs() < 1e-15,
                    "M⁻¹M[{i}][{j}] = {}",
                    product[i][j]
                );
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // the scaling is by a power of two and by 1/5, both exact here
    fn normalizing_reaches_unit_length_and_leaves_the_zero_quaternion_where_it_is() {
        // (1, 2, 2, 4) has norm exactly 5, so the expected components are exact in binary.
        let q = UnitQuaternion {
            w: 1.0,
            x: 2.0,
            y: 2.0,
            z: 4.0,
        }
        .normalized();
        let fifth = 1.0 / 5.0;
        assert_eq!(
            (q.w, q.x, q.y, q.z),
            (fifth, 2.0 * fifth, 2.0 * fifth, 4.0 * fifth)
        );

        let zero = UnitQuaternion {
            w: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
        .normalized();
        assert_eq!(
            (zero.w, zero.x, zero.y, zero.z),
            (0.0, 0.0, 0.0, 0.0),
            "a zero norm has to leave the quaternion alone, not divide by it"
        );
    }

    #[test]
    fn the_exponential_matches_the_half_angle_closed_form_across_the_small_angle_threshold() {
        // exp(φ) = (cos(|φ|/2), φ̂ sin(|φ|/2)). At or below the 1e-8 threshold both sides collapse
        // to (1, φ/2) bit-for-bit — `1 − |φ|²/8` is below 2⁻⁵⁴ from 1, the renormalization divides
        // by exactly 1, and `sin h == h` at those magnitudes — so the tolerance there is zero ulps,
        // which is what makes the second-order series' coefficients observable at all. Above it the
        // two orderings of the sine and the axis scaling differ, hence two.
        for axis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]] {
            for theta in [
                0.0,
                1e-14,
                1e-12,
                1e-10,
                1e-9,
                5e-9,
                9.999e-9,
                9.999_999e-9,
                1e-8,
                1e-7,
                1e-3,
                0.5,
                2.0,
                3.0,
            ] {
                let q = UnitQuaternion::from_rotation_vector(scale(axis, theta));
                let (c, s) = (libm::cos(0.5 * theta), libm::sin(0.5 * theta));
                let tol = if theta <= 1e-8 { 0 } else { 2 };
                let at = format!("|φ| = {theta:e} about {axis:?}");
                assert_within_ulps(q.w, c, tol, &format!("w at {at}"));
                assert_within_ulps(q.x, axis[0] * s, tol, &format!("x at {at}"));
                assert_within_ulps(q.y, axis[1] * s, tol, &format!("y at {at}"));
                assert_within_ulps(q.z, axis[2] * s, tol, &format!("z at {at}"));
            }
        }
    }

    #[test]
    fn the_logarithm_takes_the_short_way_round_from_a_negative_scalar_part() {
        // q and −q are the same rotation, so the log has to flip all four components before
        // reading the angle off — flipping the scalar alone, or leaving it, walks the long way
        // round and returns a vector pointing the other way with a norm of 2π − |φ|.
        let phi = scale(AXIS, 2.5);
        let q = UnitQuaternion::from_rotation_vector(phi);
        let negated = UnitQuaternion {
            w: -q.w,
            x: -q.x,
            y: -q.y,
            z: -q.z,
        };
        assert!(negated.w < 0.0, "the fixture must reach the negated branch");
        let got = negated.to_rotation_vector();
        for i in 0..3 {
            assert!(
                (got[i] - phi[i]).abs() < 1e-15,
                "component {i}: got {}, want {}",
                got[i],
                phi[i]
            );
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // exactly zero, and being exactly it is the assertion
    fn the_logarithm_round_trips_the_exponential_and_sends_the_identity_to_zero() {
        assert_eq!(
            UnitQuaternion::identity().to_rotation_vector(),
            [0.0; 3],
            "a zero vector part must take the series branch, not divide an angle by its own sine"
        );
        for theta in [1e-12, 1e-9, 1e-8, 1e-4, 0.5, 2.0, 3.0] {
            let phi = scale(AXIS, theta);
            let got = UnitQuaternion::from_rotation_vector(phi).to_rotation_vector();
            for i in 0..3 {
                assert!(
                    (got[i] - phi[i]).abs() <= 1e-15 * theta,
                    "|φ| = {theta:e}, component {i}: got {}, want {}",
                    got[i],
                    phi[i]
                );
            }
        }
    }
}
