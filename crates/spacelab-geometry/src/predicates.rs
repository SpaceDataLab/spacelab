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
//!
//! Gate: `derivations/exact_predicates.mac` shows that each exact-path expansion and each filtered
//! form is the defining determinant, that the signs mean what the docs below say, and that
//! `FILTER = 2⁻⁴⁶` covers the 4, 11 and 8 roundings of the three filters in exact rationals; it
//! also signs 36 adversarial near-degenerate inputs exactly, rebuilt by
//! `derivations/exact_predicates.gp`. The module tests
//! `the_signs_on_adversarial_inputs_are_the_exact_rational_signs` (G1) and
//! `the_filter_is_the_power_of_two_the_bound_was_proved_for` (G3) tie the code to both.

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

    /// (G1) The three predicates against the exact rational sign of their defining determinant, on
    /// the 36 adversarial configurations `derivations/exact_predicates.mac` builds and
    /// `derivations/exact_predicates.gp` rebuilds: doubles rounded onto a line, a circle or a plane
    /// at scales from `2⁻⁴⁰` to `2⁴⁰`, and three points of the Kettner et al. grid. Every one has
    /// `|det|/permanent` below `2⁻⁴⁶` (at most `1.7e-15`), so none is decided by the filter and the
    /// table reaches the exact path every time. Each coordinate is an integer mantissa and a binary
    /// exponent, so the doubles here are the derivation's to the bit.
    #[test]
    #[allow(clippy::unreadable_literal, clippy::type_complexity)]
    fn the_signs_on_adversarial_inputs_are_the_exact_rational_signs() {
        #[rustfmt::skip]
        let rows: [(u8, i8, &[(i64, i32)]); 36] = [
            (1, 1, &[(79886368914291, -88), (-1849426193858071, -92), (-6296809548610607, -93), (8819607196626453, -93), (429312573345981, -88), (-4877445201493317, -91)]),
            (1, 1, &[(1487264063653593, -91), (4013214247223663, -92), (2765091821163409, -96), (-1860711642820611, -91), (-656451124249567, -88), (-4149324101284773, -89)]),
            (1, 1, &[(3611215777224425, -93), (4870437360128535, -94), (2475455664784847, -92), (7482739473119393, -95), (3773419374859329, -94), (6323815322924633, -94)]),
            (1, -1, &[(7027141558104741, -95), (6592339817961763, -94), (7696653664735467, -93), (6324853659936795, -93), (2290049746053523, -89), (5270800848715635, -91)]),
            (1, -1, &[(6487137146455281, -54), (5121234403499983, -53), (-4026947782806383, -56), (1849810789482057, -52), (8562625495941365, -55), (2351391091530059, -52)]),
            (1, 1, &[(-5931065128362281, -56), (-657542418285163, -50), (2380018822240731, -54), (-7083924105721795, -53), (-1908957716731995, -51), (1249269372868667, -53)]),
            (1, 1, &[(-6179481017139577, -54), (8024256620114541, -53), (6279387955710955, -53), (-1683367360441725, -51), (4212952902732695, -50), (-1560330869848731, -48)]),
            (1, -1, &[(4178220143246125, -52), (1021364302133003, -51), (-3112000844431263, -52), (-5840339427305937, -53), (-2678096811252243, -48), (-3746492968179597, -49)]),
            (1, 1, &[(8943403822464295, -13), (82807644665271, -7), (-2571521076324761, -12), (5084447596409951, -15), (2376423621227751, -10), (8179028524977837, -13)]),
            (1, 1, &[(4094239023075791, -18), (6452140152244545, -17), (4364445579579891, -14), (5371344900253293, -13), (7899858716991199, -14), (4823192122007031, -12)]),
            (1, -1, &[(1793359779870161, -14), (609768862316205, -10), (3731926391757885, -16), (-8923107666866115, -13), (3544164808847001, -14), (8240669074186763, -11)]),
            (1, 1, &[(2763813285700693, -14), (-5810853370221827, -13), (5564385451595437, -13), (-5948220473777827, -13), (-51472355066049, -6), (-5549078851080585, -13)]),
            (1, 1, &[(4503599627370499, -53), (4503599627370501, -53), (3, 2), (3, 2), (3, 3), (3, 3)]),
            (1, -1, &[(4503599627370501, -53), (4503599627370499, -53), (3, 2), (3, 2), (3, 3), (3, 3)]),
            (1, 0, &[(4503599627370503, -53), (4503599627370503, -53), (3, 2), (3, 2), (3, 3), (3, 3)]),
            (2, 1, &[(745458325100801, -81), (-5540206572080319, -85), (2019787405449565, -83), (-8744343375180709, -86), (2599441039766465, -83), (-1342101614694107, -82), (5908584493048941, -84), (-2432076893328975, -83)]),
            (2, 1, &[(948638917160619, -83), (5327190709548763, -83), (4598225715221355, -85), (1249753425779489, -81), (4876299329625859, -84), (2253241812609387, -82), (6581281496462687, -84), (4984529163609465, -83)]),
            (2, -1, &[(325267609016421, -81), (4947645513973887, -85), (-4890301010986529, -91), (-4942567594616883, -88), (8707258810378149, -91), (-1067977134999611, -84), (6339613138520107, -86), (-1418287288426889, -83)]),
            (2, 1, &[(2407001898130419, -81), (-7104572350175863, -84), (4802072629062561, -82), (-6728916511535707, -84), (941708594825223, -80), (-3815751901492723, -84), (6031925060535051, -83), (-1894231260963853, -82)]),
            (2, 1, &[(-5808940309343407, -55), (-7318727896023289, -53), (-1138395352295549, -52), (-7293712896746345, -53), (-6162483408896719, -54), (-3967504388742355, -52), (-1559081668541595, -52), (-8006655563280407, -53)]),
            (2, 1, &[(6785924329220379, -52), (8456176110215799, -55), (3033625966055141, -51), (4801156289506743, -53), (8980223415810271, -55), (-963128119322519, -54), (8171041340807757, -53), (-1247811904526291, -51)]),
            (2, 1, &[(-3924927262246279, -53), (-3510107396561429, -55), (-2864168004746967, -52), (-751176805196837, -52), (-5557246311172455, -54), (-7121557189792105, -54), (-2533719936045501, -53), (-2832026000353111, -53)]),
            (2, -1, &[(7723457520115585, -53), (3883782533336263, -53), (6597820964516133, -53), (1551629898166319, -52), (3469338846837405, -52), (5761769401009747, -54), (3516810491025929, -52), (1430477080599869, -52)]),
            (2, 1, &[(4760118583890063, -22), (1188160956197975, -22), (1076028811518047, -21), (-5059880338790045, -23), (5185488621358361, -23), (-5793774941728133, -23), (4785832836073557, -22), (-6130431707173563, -23)]),
            (2, 1, &[(-1378236108116121, -20), (4277299992557857, -25), (-1591011986959791, -20), (-1574542083153783, -21), (-289143776215833, -19), (-5623768594017667, -22), (8890414565951837, -26), (-3243370718567201, -22)]),
            (2, 1, &[(6344207279087609, -23), (-2706646363448391, -23), (-4836153471840529, -26), (3829401814725921, -26), (-7858677218544489, -24), (-4150740813552981, -24), (5983371939574743, -23), (-4088765948078581, -22)]),
            (2, 1, &[(-7110872965420049, -23), (1252422004207643, -20), (-8588347952340605, -23), (8023638927709317, -23), (-8345450713435825, -23), (5309403389815201, -23), (-5172529241747263, -23), (6612080012627689, -24)]),
            (3, -1, &[(-8927229941391491, -74), (-2063222941608779, -71), (8877242086872129, -75), (7546159044906313, -74), (-5423804964291349, -74), (643979886024137, -70), (4188998805594327, -73), (3255547011558835, -73), (6705034671305357, -74), (-2095287274771091, -72), (-7583280698990467, -72), (4746592094227721, -73)]),
            (3, -1, &[(-3074108630475715, -74), (-2854960428284307, -72), (-5748988257643935, -74), (-948227050527387, -70), (5624178432971449, -75), (8332680117945027, -76), (-8164674976913301, -74), (-2282978136817149, -73), (5405978676467029, -74), (7675017718233927, -74), (-5255590803659, -62), (1294030175199215, -71)]),
            (3, 1, &[(5382728837152463, -75), (6872497943649755, -73), (-1463536345184755, -71), (6581127490036233, -73), (-7871109131971955, -75), (-6979144579323767, -73), (1913934422100913, -72), (-3373257943853361, -74), (-672286581372999, -73), (-961326984432117, -70), (993657019507757, -68), (-4140155183048013, -71)]),
            (3, 1, &[(-8770089691462753, -53), (-4649082216710981, -54), (-6255335571394589, -55), (-3960087143033847, -54), (-1017682997582949, -52), (-942851409383793, -50), (6482465403559883, -53), (-3151969653180507, -52), (4299852380112283, -52), (-7178265661552175, -51), (6623761565928855, -53), (-4242776665196561, -50)]),
            (3, 1, &[(3969702729177379, -54), (-3570145676866995, -54), (2455945070776217, -52), (-8302088980700913, -53), (1395190699847489, -51), (-5248188236569057, -54), (72913823055009, -48), (6431667195329679, -53), (-1077829174834231, -50), (-7836608831592109, -52), (4979124497290871, -51), (-5694312272452087, -51)]),
            (3, 1, &[(4811385135573767, -54), (5825939879342351, -54), (5456492616695147, -53), (-885122002150315, -50), (-5601957491197387, -53), (2311236145489513, -53), (-2954328695008201, -52), (6431211506448165, -54), (-7283323757840981, -55), (-5510229727914277, -51), (-5558918338208149, -52), (-7254603111524637, -53)]),
            (3, 1, &[(-6194033920651957, -33), (-7201598151455123, -33), (-626491142134515, -31), (7502389421491337, -34), (2520746093355997, -32), (1626330029418561, -31), (-4176132315771411, -33), (-1099427725478571, -30), (4579089749182791, -33), (4652460873035351, -31), (7034609483371721, -31), (3443637707642043, -31)]),
            (3, 1, &[(2335975046252349, -33), (-8590408845613349, -33), (6351198568444047, -33), (-2331440147409329, -33), (4440072027929861, -32), (-6154557575108677, -34), (2115243179045779, -33), (-5553153422472305, -34), (-8330505314577001, -33), (-5836968079521141, -33), (5216527959834875, -31), (-6917581700633727, -33)]),
            (3, -1, &[(7367733439842683, -33), (-1619181057340965, -34), (8357787316364443, -34), (7930596338111745, -33), (-363794618531071, -29), (-1927909822307099, -31), (-1177740429186365, -31), (7456656397724795, -33), (-495610299009493, -29), (-7888214851881547, -32), (3210999180559253, -30), (1209276196228955, -30)]),
        ];
        #[allow(clippy::cast_precision_loss)]
        let double = |(mantissa, exponent): (i64, i32)| libm::ldexp(mantissa as f64, exponent);
        let p2 = |c: &[(i64, i32)], k: usize| [double(c[2 * k]), double(c[2 * k + 1])];
        let p3 = |c: &[(i64, i32)], k: usize| {
            [double(c[3 * k]), double(c[3 * k + 1]), double(c[3 * k + 2])]
        };
        for (row, &(kind, want, c)) in rows.iter().enumerate() {
            let got = match kind {
                1 => orient2d(&p2(c, 0), &p2(c, 1), &p2(c, 2)),
                2 => in_circle(&p2(c, 0), &p2(c, 1), &p2(c, 2), &p2(c, 3)),
                _ => orient3d(&p3(c, 0), &p3(c, 1), &p3(c, 2), &p3(c, 3)),
            };
            assert_eq!(got, want, "row {row} (predicate {kind})");
        }
    }

    /// (G3) The filter is the power of two `derivations/exact_predicates.mac` proves sufficient:
    /// with `u = 2⁻⁵³`, `(1+u)ᵏ − 1 ≤ 2⁻⁴⁶(1−u)ᵏ` for the `k = 4, 11, 8` roundings of orient2d,
    /// `in_circle` and `orient3d`, with margins of 32, 11.6 and 16. Being a power of two is part of the
    /// proof — `FILTER · permanent` is then exact — so the value is pinned bit for bit.
    #[test]
    fn the_filter_is_the_power_of_two_the_bound_was_proved_for() {
        assert_eq!(FILTER.to_bits(), libm::ldexp(1.0, -46).to_bits());
    }
}
