//! Three-dimensional vectors and matrices.
//!
//! # Why not `nalgebra`
//!
//! An earlier design note in `plans/01-architecture.md` chose `nalgebra` and called
//! hand-rolling this "a two-hour job that turns into a quarter of debugging
//! quaternion conventions". That reasoning does not survive contact with what we
//! actually need.
//!
//! There are no quaternions here. Every rotation in this library is built from a
//! named axis rotation with a published sign convention, and the conventions are
//! checked numerically against an oracle. The ambiguity the note feared does not
//! exist in this crate.
//!
//! What does exist is a layout constraint. The hot loops of a coverage engine want
//! structure-of-arrays and eight epochs to a SIMD lane; `nalgebra`'s
//! column-major `Matrix3` is array-of-structures and would be fought, not used.
//! Roughly a hundred lines of unambiguous arithmetic buys the freedom to lay the
//! data out later however the profiler asks.
//!
//! If eigendecompositions, SVD or quaternion interpolation ever appear, `nalgebra`
//! goes in then and this module goes away.

use core::ops::{Add, Mul, Neg, Sub};

/// A vector in three-dimensional Euclidean space.
///
/// Deliberately untagged by frame. The frame-tagged wrappers live in
/// `spacelab-frames`; this is the raw arithmetic beneath them.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Vec3 {
    /// First component.
    pub x: f64,
    /// Second component.
    pub y: f64,
    /// Third component.
    pub z: f64,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// Construct from components.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// The Euclidean inner product.
    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        // `mul_add` is std-only; libm::fma is the no_std route to a single rounding.
        libm::fma(
            self.x,
            other.x,
            libm::fma(self.y, other.y, self.z * other.z),
        )
    }

    /// The vector product, right-handed.
    #[must_use]
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// The Euclidean length.
    #[must_use]
    pub fn norm(self) -> f64 {
        libm::sqrt(self.dot(self))
    }

    /// Scale by a factor.
    #[must_use]
    pub fn scale(self, k: f64) -> Self {
        Self::new(self.x * k, self.y * k, self.z * k)
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// A 3×3 matrix, stored row-major.
///
/// Row-major because every use here is `M · v`, and a row-major layout puts each
/// dot product in one contiguous run of memory.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat3 {
    /// Rows, each of three elements.
    pub rows: [[f64; 3]; 3],
}

impl Mat3 {
    /// The identity.
    pub const IDENTITY: Self = Self {
        rows: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    /// Construct from rows.
    #[must_use]
    pub const fn from_rows(rows: [[f64; 3]; 3]) -> Self {
        Self { rows }
    }

    /// Rotation of the *coordinate frame* about the first axis, by `theta`.
    ///
    /// Equivalently, rotation of a vector by `-theta`. This is the convention of
    /// the astronomical literature (`ROT1` in Vallado, `Rx` in the IERS
    /// Conventions), and mixing it with the active convention flips signs
    /// everywhere downstream.
    #[must_use]
    pub fn rot1(theta: f64) -> Self {
        let (s, c) = (libm::sin(theta), libm::cos(theta));
        Self::from_rows([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
    }

    /// Frame rotation about the second axis.
    #[must_use]
    pub fn rot2(theta: f64) -> Self {
        let (s, c) = (libm::sin(theta), libm::cos(theta));
        Self::from_rows([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    }

    /// Frame rotation about the third axis.
    #[must_use]
    pub fn rot3(theta: f64) -> Self {
        let (s, c) = (libm::sin(theta), libm::cos(theta));
        Self::from_rows([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    }

    /// The transpose. For a rotation, also the inverse.
    #[must_use]
    pub fn transpose(self) -> Self {
        let m = self.rows;
        Self::from_rows([
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ])
    }

    /// The determinant. Exactly `+1` for a rotation, to within rounding.
    #[must_use]
    pub fn determinant(self) -> f64 {
        let m = self.rows;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        let r = self.rows;
        Vec3::new(
            r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
            r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
            r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
        )
    }
}

impl Mul for Mat3 {
    type Output = Self;

    /// `self * rhs`, in the usual sense: apply `rhs` first.
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self.rows, rhs.rows);
        let mut out = [[0.0_f64; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        Self::from_rows(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_PI_2, PI};

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn assert_orthonormal(m: Mat3) {
        let should_be_identity = m * m.transpose();
        for i in 0..3 {
            for j in 0..3 {
                let expected = f64::from(u8::from(i == j));
                assert!(
                    close(should_be_identity.rows[i][j], expected, 1e-15),
                    "R·Rᵀ is not the identity at ({i},{j}): {:?}",
                    should_be_identity.rows
                );
            }
        }
        assert!(
            close(m.determinant(), 1.0, 1e-15),
            "det = {}",
            m.determinant()
        );
    }

    #[test]
    fn axis_rotations_are_proper_rotations() {
        for theta in [0.0, 0.3, FRAC_PI_2, 2.0, PI, -1.1, 5.9] {
            assert_orthonormal(Mat3::rot1(theta));
            assert_orthonormal(Mat3::rot2(theta));
            assert_orthonormal(Mat3::rot3(theta));
        }
    }

    #[test]
    fn rot3_rotates_the_frame_not_the_vector() {
        // Rotating the frame by +90° about z sends the x-axis onto -y.
        // (Rotating the *vector* would send it onto +y. Getting this backwards
        // mirrors every ground track about the Greenwich meridian.)
        let v = Mat3::rot3(FRAC_PI_2) * Vec3::new(1.0, 0.0, 0.0);
        assert!(close(v.x, 0.0, 1e-15) && close(v.y, -1.0, 1e-15) && close(v.z, 0.0, 1e-15));
    }

    #[test]
    fn composition_applies_the_right_operand_first() {
        let a = Mat3::rot3(0.4);
        let b = Mat3::rot1(0.7);
        let v = Vec3::new(1.0, 2.0, 3.0);
        let composed = (b * a) * v;
        let sequential = b * (a * v);
        assert!(close(composed.x, sequential.x, 1e-15));
        assert!(close(composed.y, sequential.y, 1e-15));
        assert!(close(composed.z, sequential.z, 1e-15));
    }

    #[test]
    fn products_of_rotations_are_rotations() {
        let m = Mat3::rot3(1.2) * Mat3::rot2(-0.3) * Mat3::rot1(0.8);
        assert_orthonormal(m);
    }

    #[test]
    fn cross_and_dot_agree_with_the_right_hand_rule() {
        let (x, y, z) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert_eq!(x.cross(y), z);
        assert_eq!(y.cross(z), x);
        assert_eq!(z.cross(x), y);
        // Exact in binary floating point: the products are 0 and 1 with no
        // rounding anywhere. A tolerance here would weaken the claim, and a
        // tolerance of zero with a strict `<` would be a test that cannot pass.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(x.dot(y), 0.0);
            assert_eq!(z.dot(z), 1.0);
        }
        assert!(close(Vec3::new(3.0, 4.0, 0.0).norm(), 5.0, 1e-15));
    }

    #[test]
    fn cross_is_pinned_by_vectors_with_no_zero_component() {
        // The basis triad above cannot see a sign error: in `y·z' − z·y'` one factor of each
        // product is zero, so flipping the minus changes nothing. These two have no zero
        // component anywhere, and the three properties below fix the sign of every term.
        let a = Vec3::new(2.0, -3.0, 5.0);
        let b = Vec3::new(-1.0, 4.0, 7.0);
        let c = a.cross(b);
        assert_eq!(c, Vec3::new(-41.0, -19.0, 5.0));

        // Perpendicular to both operands, anticommutative, and of the length Lagrange's identity
        // demands. A `+` in place of any `−` makes the product symmetric and breaks all three.
        assert!(close(c.dot(a), 0.0, 1e-13));
        assert!(close(c.dot(b), 0.0, 1e-13));
        let reversed = b.cross(a);
        assert_eq!(reversed, Vec3::new(41.0, 19.0, -5.0));
        let lagrange = a.dot(a) * b.dot(b) - a.dot(b) * a.dot(b);
        assert!(close(c.dot(c), lagrange, 1e-12));
    }

    #[test]
    fn scale_multiplies_every_component() {
        // Distinct components and a factor that is not 1: `x * k` and `x + k` agree only when
        // `x (k − 1) = k`, which none of these three satisfies.
        let v = Vec3::new(2.0, -3.0, 0.5).scale(4.0);
        assert_eq!(v, Vec3::new(8.0, -12.0, 2.0));
        assert_eq!(
            Vec3::new(1.0, 2.0, 3.0).scale(0.0),
            Vec3::new(0.0, 0.0, 0.0)
        );
    }

    #[test]
    fn determinant_measures_more_than_rotations() {
        // Every other use of `determinant` here is on a rotation, where the answer is 1 — so a
        // determinant that ignored its argument and returned 1 would satisfy all of them. These
        // three do not: a general matrix, a reflection, and a singular one.
        let general = Mat3::from_rows([[2.0, -3.0, 1.0], [4.0, 0.0, -2.0], [-1.0, 5.0, 3.0]]);
        assert!(close(general.determinant(), 70.0, 1e-12));

        let reflection = Mat3::from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
        assert!(close(reflection.determinant(), -1.0, 1e-15));

        let singular = Mat3::from_rows([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]]);
        assert!(close(singular.determinant(), 0.0, 1e-15));
    }

    #[test]
    fn matrix_vector_product_is_the_row_dot_products() {
        // Rotations applied to basis vectors leave most terms zero, so the additions between them
        // are untested there. A dense matrix on a dense vector exercises all nine products.
        // No component of `v` is ±1, or `r * v` and `r / v` would agree on that column and the
        // multiplication there would go untested.
        let m = Mat3::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
        let v = Vec3::new(-2.0, 0.5, 4.0);
        assert_eq!(m * v, Vec3::new(11.0, 18.5, 30.0));
    }
}
