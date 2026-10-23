//! Exact geometric predicates.
//!
//! Every combinatorial decision in this crate — which side of a line, inside which circumcircle, in
//! front of which face — is the *sign* of a determinant. Computed in `f64` those signs are not merely
//! inaccurate but inconsistent: a point can be reported left of `ab` and left of `ba`, and a
//! triangulation built on contradictory answers does not fail loudly, it loops or emits a mesh that is
//! not a triangulation. That is why these return an exact sign rather than an approximate value.
//!
//! Each predicate runs a floating-point filter first and falls back to expansion arithmetic only when
//! the filter cannot certify the sign. The filter's error bound is deliberately loose: a bound that is
//! too generous costs time on the near-degenerate inputs that take the exact path, while a bound that
//! is too tight is a wrong answer. The exact path is the same determinant expanded in the *original*
//! coordinates, so no difference is ever rounded — products are split by `two_product` and sums
//! accumulated as non-overlapping expansions, both of which are exact for `f64` inputs.
//!
//! References: Shewchuk, *Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
//! Predicates*, Discrete & Computational Geometry 18(3), 1997, for the expansion arithmetic; the
//! predicates themselves are the classical determinants.

use alloc::vec::Vec;

/// `a + b` as an exact unevaluated sum: the rounded sum and the rounding error it dropped.
///
/// Knuth's two-sum. Exact for any finite `a`, `b` under round-to-nearest.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let b_virtual = x - a;
    let a_virtual = x - b_virtual;
    (x, (a - a_virtual) + (b - b_virtual))
}

/// `a · b` as an exact unevaluated sum, via the fused multiply-add: `fma(a, b, −ab)` is exactly the
/// error the rounded product dropped.
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    (x, libm::fma(a, b, -x))
}

/// A non-overlapping expansion in increasing order of magnitude: its exact value is the sum of its
/// components, and its sign is the sign of the last one.
///
/// No component is ever zero — every one of them is pushed under a `!= 0.0` test — so an expansion
/// whose value is exactly zero has no components at all. [`sign_of`] depends on that.
type Expansion = Vec<f64>;

/// `e + b`, exactly. Shewchuk's `grow_expansion` with zero elimination.
fn grow(e: &[f64], b: f64) -> Expansion {
    let mut carry = b;
    let mut out = Vec::with_capacity(e.len() + 1);
    for &component in e {
        let (sum, error) = two_sum(carry, component);
        carry = sum;
        if error != 0.0 {
            out.push(error);
        }
    }
    if carry != 0.0 {
        out.push(carry);
    }
    out
}

/// `e + f`, exactly.
///
/// `O(|e|·|f|)` rather than Shewchuk's linear merge: the expansions here never exceed a few dozen
/// components, and a correct quadratic loop is worth more than a fast one that has to be argued about.
fn add(e: &[f64], f: &[f64]) -> Expansion {
    let mut out: Expansion = e.to_vec();
    for &component in f {
        out = grow(&out, component);
    }
    out
}

/// `−e`, exactly.
fn negate(e: &[f64]) -> Expansion {
    e.iter().map(|c| -c).collect()
}

/// `e · b`, exactly. Shewchuk's `scale_expansion` with zero elimination.
fn scale(e: &[f64], b: f64) -> Expansion {
    let mut out = Vec::with_capacity(2 * e.len());
    let mut carry = 0.0;
    for (index, &component) in e.iter().enumerate() {
        let (product, error) = two_product(component, b);
        if index == 0 {
            carry = product;
            if error != 0.0 {
                out.push(error);
            }
        } else {
            let (sum, sum_error) = two_sum(carry, error);
            if sum_error != 0.0 {
                out.push(sum_error);
            }
            let (next, next_error) = two_sum(product, sum);
            if next_error != 0.0 {
                out.push(next_error);
            }
            carry = next;
        }
    }
    if carry != 0.0 {
        out.push(carry);
    }
    out
}

/// `e · f`, exactly, as the sum of `e` scaled by each component of `f`.
fn multiply(e: &[f64], f: &[f64]) -> Expansion {
    let mut out = Expansion::new();
    for &component in f {
        out = add(&out, &scale(e, component));
    }
    out
}

/// The exact sign of an expansion: `-1`, `0` or `+1`.
///
/// Zero is the empty expansion, so the loop returning without a verdict *is* the zero case.
fn sign_of(e: &[f64]) -> i8 {
    for &component in e.iter().rev() {
        if component > 0.0 {
            return 1;
        }
        if component < 0.0 {
            return -1;
        }
    }
    0
}

/// `a·d − b·c`, exactly.
fn cross(a: f64, b: f64, c: f64, d: f64) -> Expansion {
    let (high, low) = two_product(a, d);
    let (neg_high, neg_low) = two_product(b, c);
    add(&[low, high], &[-neg_low, -neg_high])
}

/// `x² + y²`, exactly: the paraboloid lift that turns "inside the circumcircle" into a determinant.
fn lift(p: &[f64; 2]) -> Expansion {
    let (x_high, x_low) = two_product(p[0], p[0]);
    let (y_high, y_low) = two_product(p[1], p[1]);
    add(&[x_low, x_high], &[y_low, y_high])
}

/// How far the floating-point filter's result must exceed the magnitude of its terms before its sign is
/// certified.
///
/// The exact bound for each predicate is a small multiple of `ε` times the *permanent* — the same
/// expression as the determinant with every subtraction replaced by an addition and every factor by its
/// absolute value. The permanent, not the determinant's own terms: a term like `adx·(bdy·cdz − bdz·cdy)`
/// can cancel to nearly nothing while the rounding committed in forming it stays proportional to
/// `|bdy·cdz| + |bdz·cdy|`, so scaling the bound by the cancelled value throws that error away. This is
/// the multiple, rounded up generously and shared by all three predicates: being too generous only sends
/// more inputs down the exact path, and being too tight is a wrong sign, which is not recoverable.
const FILTER: f64 = 64.0 * f64::EPSILON;

/// Which side of the directed line `a → b` the point `c` lies on: `+1` left (counter-clockwise turn),
/// `-1` right, `0` exactly collinear.
#[must_use]
pub fn orient2d(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> i8 {
    let left = (a[0] - c[0]) * (b[1] - c[1]);
    let right = (a[1] - c[1]) * (b[0] - c[0]);
    let determinant = left - right;
    let bound = FILTER * (libm::fabs(left) + libm::fabs(right));
    if libm::fabs(determinant) > bound {
        return if determinant > 0.0 { 1 } else { -1 };
    }
    orient2d_exact(a, b, c)
}

/// `det[[ax−cx, ay−cy], [bx−cx, by−cy]]` expanded in the original coordinates, where every product is
/// exact: `ax·by − ax·cy − ay·bx + ay·cx + bx·cy − by·cx`.
fn orient2d_exact(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> i8 {
    let terms = add(
        &add(
            &cross(a[0], a[1], b[0], b[1]),
            &cross(b[0], b[1], c[0], c[1]),
        ),
        &cross(c[0], c[1], a[0], a[1]),
    );
    sign_of(&terms)
}

/// Whether `d` lies inside the circle through `a`, `b`, `c`: `+1` inside, `-1` outside, `0` exactly on
/// it (four cocircular points).
///
/// The sign assumes `a`, `b`, `c` are counter-clockwise — [`orient2d`] positive. For a clockwise
/// triangle the sign is reversed, which is the caller's business to know: the Delaunay condition is
/// stated on oriented triangles.
#[must_use]
pub fn in_circle(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2], d: &[f64; 2]) -> i8 {
    let (adx, ady) = (a[0] - d[0], a[1] - d[1]);
    let (bdx, bdy) = (b[0] - d[0], b[1] - d[1]);
    let (cdx, cdy) = (c[0] - d[0], c[1] - d[1]);
    let (alift, blift, clift) = (
        adx * adx + ady * ady,
        bdx * bdx + bdy * bdy,
        cdx * cdx + cdy * cdy,
    );
    let bc = bdx * cdy - cdx * bdy;
    let ca = cdx * ady - adx * cdy;
    let ab = adx * bdy - bdx * ady;
    let determinant = alift * bc + blift * ca + clift * ab;
    let permanent = (libm::fabs(bdx * cdy) + libm::fabs(cdx * bdy)) * alift
        + (libm::fabs(cdx * ady) + libm::fabs(adx * cdy)) * blift
        + (libm::fabs(adx * bdy) + libm::fabs(bdx * ady)) * clift;
    if libm::fabs(determinant) > FILTER * permanent {
        return if determinant > 0.0 { 1 } else { -1 };
    }
    in_circle_exact(a, b, c, d)
}

/// The 4×4 lifted determinant expanded along its column of ones, in the original coordinates.
fn in_circle_exact(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2], d: &[f64; 2]) -> i8 {
    let total = add(
        &add(&negate(&minor3(b, c, d)), &minor3(a, c, d)),
        &add(&negate(&minor3(a, b, d)), &minor3(a, b, c)),
    );
    sign_of(&total)
}

/// `det[[px, py, |p|²], [qx, qy, |q|²], [rx, ry, |r|²]]`, exactly.
fn minor3(p: &[f64; 2], q: &[f64; 2], r: &[f64; 2]) -> Expansion {
    let (q_lift, r_lift) = (lift(q), lift(r));
    let first = scale(
        &add(&scale(&r_lift, q[1]), &negate(&scale(&q_lift, r[1]))),
        p[0],
    );
    let second = scale(
        &add(&scale(&r_lift, q[0]), &negate(&scale(&q_lift, r[0]))),
        p[1],
    );
    let third = multiply(&lift(p), &cross(q[0], q[1], r[0], r[1]));
    add(&add(&first, &negate(&second)), &third)
}

/// Which side of the plane through `a`, `b`, `c` the point `d` lies on: the sign of
/// `(a − d) · ((b − d) × (c − d))`.
///
/// `+1` means `d` is on the side the triangle's own normal `(b − a) × (c − a)` points *away* from —
/// behind the face, for an outward-oriented hull face. `-1` means in front of it, `0` exactly coplanar.
#[must_use]
pub fn orient3d(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3]) -> i8 {
    let (adx, ady, adz) = (a[0] - d[0], a[1] - d[1], a[2] - d[2]);
    let (bdx, bdy, bdz) = (b[0] - d[0], b[1] - d[1], b[2] - d[2]);
    let (cdx, cdy, cdz) = (c[0] - d[0], c[1] - d[1], c[2] - d[2]);
    let bc = bdy * cdz - bdz * cdy;
    let ca = bdz * cdx - bdx * cdz;
    let ab = bdx * cdy - bdy * cdx;
    let determinant = adx * bc + ady * ca + adz * ab;
    let permanent = (libm::fabs(bdy * cdz) + libm::fabs(bdz * cdy)) * libm::fabs(adx)
        + (libm::fabs(bdz * cdx) + libm::fabs(bdx * cdz)) * libm::fabs(ady)
        + (libm::fabs(bdx * cdy) + libm::fabs(bdy * cdx)) * libm::fabs(adz);
    if libm::fabs(determinant) > FILTER * permanent {
        return if determinant > 0.0 { 1 } else { -1 };
    }
    orient3d_exact(a, b, c, d)
}

/// The 4×4 determinant with a column of ones, expanded along that column in the original coordinates.
fn orient3d_exact(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3]) -> i8 {
    let total = add(
        &add(&negate(&minor3d(b, c, d)), &minor3d(a, c, d)),
        &add(&negate(&minor3d(a, b, d)), &minor3d(a, b, c)),
    );
    sign_of(&total)
}

/// `det[[px, py, pz], [qx, qy, qz], [rx, ry, rz]]`, exactly.
fn minor3d(p: &[f64; 3], q: &[f64; 3], r: &[f64; 3]) -> Expansion {
    let first = scale(&cross(q[1], q[2], r[1], r[2]), p[0]);
    let second = scale(&cross(q[0], q[2], r[0], r[2]), p[1]);
    let third = scale(&cross(q[0], q[1], r[0], r[1]), p[2]);
    add(&add(&first, &negate(&second)), &third)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]
    use super::*;
    use core::cmp::Ordering;

    #[test]
    fn the_turn_of_a_triangle_is_its_orientation() {
        let (a, b, c) = ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]);
        assert_eq!(orient2d(&a, &b, &c), 1);
        assert_eq!(orient2d(&a, &c, &b), -1);
        assert_eq!(orient2d(&a, &b, &[2.0, 0.0]), 0);
    }

    /// The filter must never certify a sign the exact computation contradicts. These points are
    /// collinear by construction but their `f64` determinant is not zero — `(0.5, 0.5)` and
    /// `(12, 12)` are on the diagonal, and the third point is one ulp off it.
    #[test]
    fn a_determinant_below_its_own_rounding_error_still_gets_the_right_sign() {
        let a = [0.5, 0.5];
        let b = [12.0, 12.0];
        let on = [17.3, 17.3];
        assert_eq!(orient2d(&a, &b, &on), 0);

        let above = [17.3, 17.3 + f64::EPSILON * 17.3];
        let below = [17.3, 17.3 - f64::EPSILON * 17.3];
        assert_eq!(orient2d(&a, &b, &above), 1);
        assert_eq!(orient2d(&a, &b, &below), -1);
    }

    /// Anti-symmetry is the property a filter without an exact fallback breaks: swapping two arguments
    /// must flip the sign for *every* input, including the ones the filter cannot certify.
    #[test]
    fn swapping_two_points_flips_every_sign() {
        let mut seed = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            // A coarse grid makes exact degeneracies common rather than measure-zero.
            #[allow(clippy::cast_precision_loss)]
            let value = ((seed >> 40) % 7) as f64;
            value
        };
        for _ in 0..2000 {
            let a = [next(), next()];
            let b = [next(), next()];
            let c = [next(), next()];
            let d = [next(), next()];
            assert_eq!(orient2d(&a, &b, &c), -orient2d(&b, &a, &c));
            if orient2d(&a, &b, &c) == 1 {
                // in_circle is anti-symmetric in any transposition, and the cocircular case is where
                // the filter is guaranteed to abstain.
                assert_eq!(in_circle(&a, &b, &c, &d), -in_circle(&b, &a, &c, &d));
            }
        }
    }

    #[test]
    fn the_circumcircle_of_the_unit_square_has_its_corners_exactly_on_it() {
        let (a, b, c) = ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0]);
        assert_eq!(in_circle(&a, &b, &c, &[0.0, 1.0]), 0);
        assert_eq!(in_circle(&a, &b, &c, &[0.5, 0.5]), 1);
        assert_eq!(in_circle(&a, &b, &c, &[-1.0, 2.0]), -1);
    }

    /// Kettner, Mehlhorn, Pion, Schirra & Yap, *Classroom examples of robustness problems in geometric
    /// computations*, Computational Geometry 40(1), 2008: the grid `(0.5 + i·2⁻⁵³, 0.5 + j·2⁻⁵³)`
    /// against the line through `(12, 12)` and `(24, 24)`. Expanded, the determinant is exactly
    /// `12·2⁻⁵³·(j − i)`, so the answer is `sign(j − i)` and nothing else — while the `f64` determinant
    /// is the difference of two products of order 270 and is dominated by their rounding. The count at
    /// the end is what makes this a test of the exact path rather than of the filter: it fails if the
    /// fixture ever stops being one floating point cannot do.
    #[test]
    fn the_classroom_example_of_a_wrong_orientation_comes_out_right() {
        let ulp = f64::EPSILON / 2.0; // 2⁻⁵³, one ulp at 0.5
        let (q, r) = ([12.0, 12.0], [24.0, 24.0]);
        let mut naive_wrong = 0;
        for i in 0..16 {
            for j in 0..16 {
                let p = [0.5 + f64::from(i) * ulp, 0.5 + f64::from(j) * ulp];
                let expected: i8 = match j.cmp(&i) {
                    Ordering::Greater => 1,
                    Ordering::Less => -1,
                    Ordering::Equal => 0,
                };
                assert_eq!(orient2d(&p, &q, &r), expected, "i = {i}, j = {j}");

                let naive = (p[0] - r[0]) * (q[1] - r[1]) - (p[1] - r[1]) * (q[0] - r[0]);
                let naive_sign: i8 = if naive > 0.0 {
                    1
                } else if naive < 0.0 {
                    -1
                } else {
                    0
                };
                if naive_sign != expected {
                    naive_wrong += 1;
                }
            }
        }
        assert!(
            naive_wrong > 0,
            "the fixture no longer exercises anything the filter cannot certify"
        );
    }

    /// One ulp decides a Delaunay edge. The circle through `(0, 0)`, `(1, 0)`, `(1, 1)` passes exactly
    /// through `(0, 1)`, and moving the fourth point in `y` moves it away from the centre `(0.5, 0.5)`
    /// in both directions — so a larger `y` is strictly outside and a smaller one strictly inside,
    /// however small the step and whatever the rounding of products of order one says.
    #[test]
    fn one_ulp_decides_whether_a_point_is_inside_the_circumcircle() {
        let (a, b, c) = ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0]);
        assert_eq!(orient2d(&a, &b, &c), 1, "the sign convention assumes CCW");
        for step in 1..=8_u64 {
            let above = f64::from_bits(1.0_f64.to_bits() + step);
            let below = f64::from_bits(1.0_f64.to_bits() - step);
            assert_eq!(in_circle(&a, &b, &c, &[0.0, above]), -1, "step = {step}");
            assert_eq!(in_circle(&a, &b, &c, &[0.0, below]), 1, "step = {step}");
        }
    }

    /// The plane `x + y + z = 1.5` through `(1.5, 0, 0)`, `(0, 1.5, 0)` and `(0, 0, 1.5)`, against
    /// points a few ulps either side of its centre. The determinant is exactly `2.25·2⁻⁵³·(i + j + k)`,
    /// which is fifteen orders of magnitude below the terms it is the difference of.
    #[test]
    fn a_point_one_ulp_off_a_plane_is_not_on_it() {
        let ulp = f64::EPSILON / 2.0;
        let (b, c, d) = ([1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]);
        for i in -2..=2 {
            for j in -2..=2 {
                for k in -2..=2 {
                    let a = [
                        0.5 + f64::from(i) * ulp,
                        0.5 + f64::from(j) * ulp,
                        0.5 + f64::from(k) * ulp,
                    ];
                    let sum = i + j + k;
                    let expected: i8 = match sum.cmp(&0) {
                        Ordering::Greater => 1,
                        Ordering::Less => -1,
                        Ordering::Equal => 0,
                    };
                    assert_eq!(orient3d(&a, &b, &c, &d), expected, "({i}, {j}, {k})");
                }
            }
        }
    }

    /// A deterministic stream of doubles in `[-1, 1)` with every mantissa bit populated. The fixtures
    /// below need coordinates whose *differences* round as well as whose products do, which a coarse
    /// grid cannot give: on a grid the subtractions are exact and the only reachable failure is a
    /// determinant that collapses to zero.
    fn uniform(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        #[allow(clippy::cast_precision_loss)]
        let unit = (*state >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0); // 2⁻⁵³
        2.0f64.mul_add(unit, -1.0)
    }

    fn point(state: &mut u64) -> [f64; 3] {
        [uniform(state), uniform(state), uniform(state)]
    }

    /// A power of two spanning `2⁻²⁶` to `2²⁶`. Scaling every coordinate by a power of two is exact, so
    /// it leaves the rounding structure of a configuration alone and changes only the sizes the error
    /// estimate has to combine — which is the axis an estimate that mixes them wrongly fails on.
    fn scale(state: &mut u64) -> f64 {
        #[allow(clippy::cast_possible_truncation)]
        let exponent = (uniform(state) * 26.0) as i32;
        2.0f64.powi(exponent)
    }

    /// A separation of `2⁻³⁰` to `2⁻¹⁰` of the configuration's own size: near-duplicate points, which a
    /// mesh meets constantly and a uniform sample never does.
    fn crowd(state: &mut u64) -> f64 {
        #[allow(clippy::cast_possible_truncation)]
        let exponent = -10 - (libm::fabs(uniform(state)) * 20.0) as i32;
        2.0f64.powi(exponent)
    }

    fn sign_of_f64(value: f64) -> i8 {
        match value.partial_cmp(&0.0) {
            Some(Ordering::Greater) => 1,
            Some(Ordering::Less) => -1,
            _ => 0,
        }
    }

    /// The filter is a shortcut, never an answer: wherever it certifies a sign, the exact path must
    /// agree. The two can only differ within rounding of a degeneracy, so that is what this samples —
    /// a point placed on the line through the other two in `f64`, which lands it a rounding error off
    /// the real one. The count at the end is the fixture's own self-check: it fails if the samples stop
    /// reaching the regime where the naive determinant is wrong.
    #[test]
    fn the_orientation_filter_never_certifies_what_the_exact_path_denies() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut naive_wrong = 0;
        for _ in 0..20_000 {
            let a = [uniform(&mut state), uniform(&mut state)];
            let b = [uniform(&mut state), uniform(&mut state)];
            let t = uniform(&mut state) * 3.0;
            let c = [t.mul_add(b[0] - a[0], a[0]), t.mul_add(b[1] - a[1], a[1])];
            let exact = orient2d_exact(&a, &b, &c);
            assert_eq!(
                orient2d(&a, &b, &c),
                exact,
                "a = {a:?}, b = {b:?}, c = {c:?}"
            );
            let naive = (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]);
            if sign_of_f64(naive) != exact {
                naive_wrong += 1;
            }
        }
        assert!(
            naive_wrong > 100,
            "only {naive_wrong} samples needed the exact path"
        );
    }

    /// The same property for the circumcircle, sampled by putting all four points on one circle: their
    /// `f64` coordinates then sit a rounding error off it, in either direction. The circle's size runs
    /// over fifty orders of magnitude and half the samples crowd two points together, because an error
    /// estimate that combines its terms wrongly is only wrong where the terms differ in size.
    #[test]
    fn the_circumcircle_filter_never_certifies_what_the_exact_path_denies() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut naive_wrong = 0;
        for _ in 0..20_000 {
            let size = scale(&mut state);
            let centre = [uniform(&mut state) * size, uniform(&mut state) * size];
            let radius = uniform(&mut state).mul_add(0.75, 1.25) * size;
            let mut angles = [0.0; 4];
            for angle in &mut angles {
                *angle = uniform(&mut state) * core::f64::consts::PI;
            }
            if uniform(&mut state) > 0.0 {
                angles[3] = angles[2] + crowd(&mut state);
            }
            if uniform(&mut state) > 0.0 {
                angles[1] = angles[0] + crowd(&mut state);
            }
            let on_circle = |angle: f64| {
                [
                    radius.mul_add(libm::cos(angle), centre[0]),
                    radius.mul_add(libm::sin(angle), centre[1]),
                ]
            };
            let (a, b, c, d) = (
                on_circle(angles[0]),
                on_circle(angles[1]),
                on_circle(angles[2]),
                on_circle(angles[3]),
            );
            let exact = in_circle_exact(&a, &b, &c, &d);
            assert_eq!(
                in_circle(&a, &b, &c, &d),
                exact,
                "a = {a:?}, b = {b:?}, c = {c:?}, d = {d:?}"
            );
            let (adx, ady) = (a[0] - d[0], a[1] - d[1]);
            let (bdx, bdy) = (b[0] - d[0], b[1] - d[1]);
            let (cdx, cdy) = (c[0] - d[0], c[1] - d[1]);
            let naive = adx.mul_add(adx, ady * ady) * (bdx * cdy - cdx * bdy)
                + bdx.mul_add(bdx, bdy * bdy) * (cdx * ady - adx * cdy)
                + cdx.mul_add(cdx, cdy * cdy) * (adx * bdy - bdx * ady);
            if sign_of_f64(naive) != exact {
                naive_wrong += 1;
            }
        }
        assert!(
            naive_wrong > 100,
            "only {naive_wrong} samples needed the exact path"
        );
    }

    /// The same property in three dimensions, sampled by placing the fourth point in the plane of the
    /// other three. This is the one that caught a bound too small to be one: scaling the error estimate
    /// by `|adx·(bdy·cdz − bdz·cdy)|` throws away the rounding committed inside the bracket, which
    /// survives the cancellation the bracket does not.
    #[test]
    fn the_plane_filter_never_certifies_what_the_exact_path_denies() {
        let mut state = 0x853C_49E6_748F_EA9B_u64;
        let mut naive_wrong = 0;
        for _ in 0..20_000 {
            let size = scale(&mut state);
            let (a, mut b, mut c) = (point(&mut state), point(&mut state), point(&mut state));
            if uniform(&mut state) > 0.0 {
                let gap = crowd(&mut state);
                for axis in 0..3 {
                    c[axis] = gap.mul_add(c[axis] - b[axis], b[axis]);
                }
            }
            let a = [a[0] * size, a[1] * size, a[2] * size];
            for axis in 0..3 {
                b[axis] *= size;
                c[axis] *= size;
            }
            let (along, across) = (uniform(&mut state) * 2.0, uniform(&mut state) * 2.0);
            let mut d = [0.0; 3];
            for axis in 0..3 {
                d[axis] =
                    across.mul_add(c[axis] - a[axis], along.mul_add(b[axis] - a[axis], a[axis]));
            }
            let exact = orient3d_exact(&a, &b, &c, &d);
            assert_eq!(
                orient3d(&a, &b, &c, &d),
                exact,
                "a = {a:?}, b = {b:?}, c = {c:?}, d = {d:?}"
            );
            let (adx, ady, adz) = (a[0] - d[0], a[1] - d[1], a[2] - d[2]);
            let (bdx, bdy, bdz) = (b[0] - d[0], b[1] - d[1], b[2] - d[2]);
            let (cdx, cdy, cdz) = (c[0] - d[0], c[1] - d[1], c[2] - d[2]);
            let naive = adx * (bdy * cdz - bdz * cdy)
                + ady * (bdz * cdx - bdx * cdz)
                + adz * (bdx * cdy - bdy * cdx);
            if sign_of_f64(naive) != exact {
                naive_wrong += 1;
            }
        }
        assert!(
            naive_wrong > 100,
            "only {naive_wrong} samples needed the exact path"
        );
    }

    /// The counterexample that retired the old error bound. `d` is in the plane of `a`, `b` and `c` to
    /// within a rounding error and on the positive side of it; the `f64` determinant is `-1.18e-17`,
    /// while the bound built from the cancelled sub-determinants was `2.47e-18` — small enough to
    /// certify the wrong sign. The expected value is from an independent exact evaluation over the
    /// rationals (Python `fractions.Fraction`), not from this crate's own exact path.
    #[test]
    fn a_plane_determinant_lost_to_cancellation_is_not_certified() {
        let a = [
            0.401_868_441_524_189_55,
            -0.109_750_168_206_454_11,
            0.158_328_463_236_246_05,
        ];
        let b = [
            -0.184_306_758_090_561_96,
            0.166_128_116_306_372_8,
            -0.834_146_079_808_576_2,
        ];
        let c = [
            0.736_335_224_247_804_5,
            -0.495_717_461_488_489_83,
            -0.406_297_603_507_437_93,
        ];
        let d = [
            0.406_986_275_030_167_16,
            -0.259_077_548_009_523_65,
            -0.559_986_977_498_892_9,
        ];
        assert_eq!(orient3d(&a, &b, &c, &d), 1);
    }

    #[test]
    fn a_point_behind_a_face_is_positive_and_one_in_front_is_negative() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        // The face normal (b−a)×(c−a) is +z, so +z is "in front".
        assert_eq!(orient3d(&a, &b, &c, &[0.0, 0.0, -1.0]), 1);
        assert_eq!(orient3d(&a, &b, &c, &[0.0, 0.0, 1.0]), -1);
        assert_eq!(orient3d(&a, &b, &c, &[3.0, 4.0, 0.0]), 0);
    }

    /// A coplanarity that `f64` cannot see: the plane `z = x/3 + y/7` evaluated at points whose
    /// coordinates make the division inexact.
    #[test]
    fn coplanarity_survives_coordinates_that_are_not_representable() {
        let plane = |x: f64, y: f64| [x, y, x / 3.0 + y / 7.0];
        let a = plane(1.0, 2.0);
        let b = plane(-5.0, 3.0);
        let c = plane(11.0, -13.0);
        let d = plane(7.0, 17.0);
        // Not exactly coplanar — the rounding of x/3 + y/7 puts each point a fraction of an ulp off
        // the plane — so the honest statement is that the sign is consistent under permutation.
        assert_eq!(orient3d(&a, &b, &c, &d), -orient3d(&b, &a, &c, &d));
        assert_eq!(orient3d(&a, &b, &c, &d), orient3d(&b, &c, &a, &d));
    }
}
