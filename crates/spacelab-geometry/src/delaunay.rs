//! Delaunay triangulation in the plane, by Bowyer–Watson insertion.
//!
//! The triangulation is Delaunay at every step, so inserting a point means deleting the triangles
//! whose circumcircles contain it — a connected *cavity* — and coning the cavity's boundary to the new
//! point. Nothing is retriangulated afterwards and no edge is ever flipped.
//!
//! **The outside is triangulated too.** Every convex-hull edge carries a *ghost* triangle whose third
//! vertex is the point at infinity, and the circumcircle of such a triangle is the half-plane outside
//! that edge. This is not an optimisation: it makes the mesh a closed surface, so every directed edge
//! has exactly one twin, the cavity search never runs off an unmatched boundary, and a point outside
//! the current hull is inserted by the same code as a point inside it. The alternative — a
//! "super-triangle" of far-away fake vertices, deleted at the end — is the common textbook shortcut and
//! it is not correct: how far away is far enough depends on the data, and when it is not far enough the
//! failure is a silently wrong triangulation near the hull.
//!
//! Two degeneracies are handled explicitly rather than perturbed away, because both are the *common*
//! case for gridded data rather than a measure-zero curiosity:
//!
//! * **Cocircular points** — a regular grid has four cocircular points on every cell. The in-circle
//!   test is exact, so those points are strictly outside each other's circumcircles and the resulting
//!   triangulation is one of the several valid Delaunay triangulations of the set, chosen by insertion
//!   order rather than by rounding.
//! * **A point exactly on a hull edge** — the row of a grid. A ghost triangle conflicts with a point on
//!   its own edge's line only when that point lies *between* the edge's endpoints; that test is a
//!   coordinate comparison, exact by construction. Without it the cone would build a zero-area triangle
//!   from three collinear points; with a non-strict half-plane test instead, a point on the *extension*
//!   of a hull edge would build one.
//!
//! Gate: `derivations/hull_delaunay.mac` finds the Delaunay triangulation of 14 integer points in
//! general position by brute force — every counter-clockwise triple with no other point strictly
//! inside its circumcircle, by the exact lifted determinant — and checks the `2n − 2 − h` count
//! Euler's formula gives; `derivations/hull_delaunay.gp` repeats the search. The module test
//! `the_triangulation_is_the_brute_force_delaunay_set` (G2) compares the insertion's output with it.

use crate::GeometryError;
use crate::predicates::{in_circle, orient2d};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

/// The point at infinity: the third vertex of every ghost triangle.
const GHOST: usize = usize::MAX;

/// A Delaunay triangulation of a planar point set.
///
/// Indices in [`triangles`](Self::triangles) refer to the caller's original slice. Every distinct input
/// point is a vertex, including points that lie on the convex-hull boundary; exact duplicates are
/// represented by their first occurrence.
#[derive(Debug, Clone, PartialEq)]
pub struct Triangulation {
    points: Vec<[f64; 2]>,
    triangles: Vec<[usize; 3]>,
}

impl Triangulation {
    /// Triangulate a point set.
    ///
    /// # Errors
    ///
    /// [`GeometryError::NonFiniteCoordinate`], [`GeometryError::TooFewPoints`] for fewer than three
    /// distinct points, or [`GeometryError::Degenerate`] when they are all collinear.
    pub fn new(points: &[[f64; 2]]) -> Result<Self, GeometryError> {
        for (index, p) in points.iter().enumerate() {
            if !p[0].is_finite() || !p[1].is_finite() {
                return Err(GeometryError::NonFiniteCoordinate { point: index });
            }
        }
        let unique = distinct(points);
        if unique.len() < 3 {
            return Err(GeometryError::TooFewPoints(unique.len()));
        }
        let [a, b, c] = seed_triangle(points, &unique)?;
        let mut mesh = Mesh::seed(points, a, b, c);
        for &index in &unique {
            if index != a && index != b && index != c {
                mesh.insert(index);
            }
        }
        Ok(Self {
            points: points.to_vec(),
            triangles: mesh.harvest(),
        })
    }

    /// The triangles, each counter-clockwise, as indices into the original point slice.
    #[must_use]
    pub fn triangles(&self) -> &[[usize; 3]] {
        &self.triangles
    }

    /// The points the triangulation was built from.
    #[must_use]
    pub fn points(&self) -> &[[f64; 2]] {
        &self.points
    }
}

/// The mesh under construction: triangles, a directed-edge index, and a walk hint.
///
/// The directed-edge map replaces an explicit neighbour array. Every directed edge `(u, v)` is owned by
/// exactly one live triangle, so the neighbour across it is whoever owns `(v, u)` — one `BTreeMap`
/// lookup, and one invariant to maintain instead of three.
struct Mesh<'a> {
    points: &'a [[f64; 2]],
    triangles: Vec<[usize; 3]>,
    dead: Vec<bool>,
    owner: BTreeMap<(usize, usize), usize>,
    hint: usize,
}

impl<'a> Mesh<'a> {
    /// One counter-clockwise triangle and the three ghosts that close it into a sphere.
    fn seed(points: &'a [[f64; 2]], a: usize, b: usize, c: usize) -> Self {
        let (a, b, c) = if orient2d(&points[a], &points[b], &points[c]) > 0 {
            (a, b, c)
        } else {
            (a, c, b)
        };
        let mut mesh = Self {
            points,
            triangles: Vec::new(),
            dead: Vec::new(),
            owner: BTreeMap::new(),
            hint: 0,
        };
        mesh.add([a, b, c]);
        mesh.add([b, a, GHOST]);
        mesh.add([c, b, GHOST]);
        mesh.add([a, c, GHOST]);
        mesh
    }

    /// Register a triangle, rotating the ghost vertex last so a ghost's own edge is always `(v0, v1)`.
    fn add(&mut self, triangle: [usize; 3]) -> usize {
        let rotated = match triangle.iter().position(|&v| v == GHOST) {
            Some(0) => [triangle[1], triangle[2], GHOST],
            Some(1) => [triangle[2], triangle[0], GHOST],
            _ => triangle,
        };
        let index = self.triangles.len();
        self.triangles.push(rotated);
        self.dead.push(false);
        for k in 0..3 {
            self.owner.insert((rotated[k], rotated[(k + 1) % 3]), index);
        }
        if rotated[2] != GHOST {
            self.hint = index;
        }
        index
    }

    fn kill(&mut self, index: usize) {
        let triangle = self.triangles[index];
        self.dead[index] = true;
        for k in 0..3 {
            self.owner.remove(&(triangle[k], triangle[(k + 1) % 3]));
        }
    }

    /// Whether the point is strictly inside triangle `index`'s circumcircle — for a ghost, strictly
    /// outside its hull edge, or exactly on that edge and between its endpoints.
    fn conflicts(&self, index: usize, point: usize) -> bool {
        let p = &self.points[point];
        let v = self.triangles[index];
        if v[2] == GHOST {
            let (a, b) = (&self.points[v[0]], &self.points[v[1]]);
            return match orient2d(a, b, p) {
                1 => true,
                0 => between(a, b, p),
                _ => false,
            };
        }
        in_circle(
            &self.points[v[0]],
            &self.points[v[1]],
            &self.points[v[2]],
            p,
        ) > 0
    }

    /// The triangle a visibility walk from the hint lands on.
    ///
    /// Step across any edge the point lies strictly to the right of. The walk terminates on a Delaunay
    /// triangulation (Devillers, Pion & Teillaud, *Walking in a triangulation*, 2001) and this mesh is
    /// Delaunay at every step, so it lands on a triangle the point conflicts with. Every triangle it
    /// visits is live: the hint is a live real triangle by construction and the owner map holds only
    /// live triangles. The step cap is there because a library that can loop forever on unexpected input
    /// is worse than one that is slow on it.
    fn walk(&self, point: usize) -> usize {
        let mut current = self.hint;
        for _ in 0..self.triangles.len() {
            let v = self.triangles[current];
            if v[2] == GHOST {
                break;
            }
            let mut stepped = false;
            for k in 0..3 {
                let (a, b) = (v[k], v[(k + 1) % 3]);
                if orient2d(&self.points[a], &self.points[b], &self.points[point]) < 0 {
                    current = self.owner[&(b, a)];
                    stepped = true;
                    break;
                }
            }
            if !stepped {
                break;
            }
        }
        current
    }

    /// The first live triangle whose circumcircle contains the point, by exhaustive search.
    fn scan(&self, point: usize) -> Option<usize> {
        (0..self.triangles.len()).find(|&index| !self.dead[index] && self.conflicts(index, point))
    }

    /// A triangle whose circumcircle contains the point, or `None` when none does — which after
    /// de-duplication means the point is already a vertex of the mesh.
    ///
    /// The walk is the fast path; the exhaustive scan behind it is what makes a walk that lands wrong
    /// slow rather than incorrect.
    fn locate(&self, point: usize) -> Option<usize> {
        let landed = self.walk(point);
        if self.conflicts(landed, point) {
            return Some(landed);
        }
        self.scan(point)
    }

    /// Insert one point: delete the cavity it conflicts with, cone the cavity's boundary to it.
    fn insert(&mut self, point: usize) {
        let Some(start) = self.locate(point) else {
            return;
        };

        let mut cavity = alloc::vec![start];
        let mut visited: BTreeSet<usize> = BTreeSet::new();
        visited.insert(start);
        let mut boundary: Vec<(usize, usize)> = Vec::new();
        let mut stack = alloc::vec![start];
        while let Some(index) = stack.pop() {
            let v = self.triangles[index];
            for k in 0..3 {
                let (a, b) = (v[k], v[(k + 1) % 3]);
                let across = self.owner[&(b, a)];
                if visited.contains(&across) {
                    continue;
                }
                if self.conflicts(across, point) {
                    visited.insert(across);
                    cavity.push(across);
                    stack.push(across);
                } else {
                    boundary.push((a, b));
                }
            }
        }

        for index in cavity {
            self.kill(index);
        }
        for (a, b) in boundary {
            self.add([a, b, point]);
        }
    }

    /// The live, non-ghost triangles.
    fn harvest(&self) -> Vec<[usize; 3]> {
        self.triangles
            .iter()
            .zip(&self.dead)
            .filter(|(triangle, dead)| !**dead && triangle[2] != GHOST)
            .map(|(triangle, _)| *triangle)
            .collect()
    }
}

/// The first two distinct points and the first one not collinear with them.
///
/// `unique` must hold at least three indices, which its caller has already checked.
fn seed_triangle(points: &[[f64; 2]], unique: &[usize]) -> Result<[usize; 3], GeometryError> {
    let (a, b) = (unique[0], unique[1]);
    let c = unique[2..]
        .iter()
        .copied()
        .find(|&i| orient2d(&points[a], &points[b], &points[i]) != 0)
        .ok_or(GeometryError::Degenerate)?;
    Ok([a, b, c])
}

/// Whether `p` lies strictly between `a` and `b`, for three points already known to be collinear.
///
/// Comparisons only — no arithmetic, so no rounding. The axis is chosen where the segment is not
/// degenerate, which for collinear points is any axis where the endpoints differ.
#[allow(clippy::float_cmp)] // exactly the point: the axis is degenerate only under exact equality
fn between(a: &[f64; 2], b: &[f64; 2], p: &[f64; 2]) -> bool {
    let strictly_inside = |lo: f64, hi: f64, x: f64| {
        if lo <= hi {
            lo < x && x < hi
        } else {
            hi < x && x < lo
        }
    };
    if a[0] == b[0] {
        strictly_inside(a[1], b[1], p[1])
    } else {
        strictly_inside(a[0], b[0], p[0])
    }
}

/// Indices of the distinct points, first occurrence winning.
fn distinct(points: &[[f64; 2]]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&i, &j| {
        points[i][0]
            .total_cmp(&points[j][0])
            .then(points[i][1].total_cmp(&points[j][1]))
    });
    // Exact equality is the intent: only a bit-identical duplicate is a duplicate.
    #[allow(clippy::float_cmp)]
    order.dedup_by(|&mut i, &mut j| points[i] == points[j]);
    order.sort_unstable();
    order
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::float_cmp)]
    use super::*;
    use crate::hull::convex_hull_2d;

    /// The defining property, checked against every point rather than sampled: no point lies strictly
    /// inside any triangle's circumcircle.
    fn assert_is_delaunay(mesh: &Triangulation) {
        for triangle in mesh.triangles() {
            let (a, b, c) = (
                &mesh.points()[triangle[0]],
                &mesh.points()[triangle[1]],
                &mesh.points()[triangle[2]],
            );
            assert_eq!(orient2d(a, b, c), 1, "triangle {triangle:?} is not CCW");
            for (index, p) in mesh.points().iter().enumerate() {
                if triangle.contains(&index) {
                    continue;
                }
                assert!(
                    in_circle(a, b, c, p) <= 0,
                    "point {index} is inside the circumcircle of {triangle:?}"
                );
            }
        }
    }

    /// Every distinct point is a vertex, and the triangle count is the one Euler's formula demands:
    /// `2n − 2 − h` for `n` vertices of which `h` are on the hull boundary. `h` is counted from the mesh
    /// itself — an edge with only one adjacent triangle — so the check does not assume the hull.
    fn assert_covers_every_point(mesh: &Triangulation) {
        let used: BTreeSet<usize> = mesh.triangles().iter().flatten().copied().collect();
        let expected: BTreeSet<usize> = distinct(mesh.points()).into_iter().collect();
        assert_eq!(used, expected, "not every distinct point is a vertex");

        let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
        for triangle in mesh.triangles() {
            for k in 0..3 {
                edges.insert((triangle[k], triangle[(k + 1) % 3]));
            }
        }
        let boundary = edges
            .iter()
            .filter(|&&(u, v)| !edges.contains(&(v, u)))
            .count();
        assert_eq!(
            mesh.triangles().len(),
            2 * used.len() - 2 - boundary,
            "triangle count contradicts Euler's formula"
        );
    }

    /// The invariants the mesh maintains for itself. Nothing the public API returns can see them: a
    /// stale edge map or a hint left on a dead triangle changes which triangles are visited, not which
    /// triangulation comes out.
    fn assert_internally_consistent(mesh: &Mesh<'_>) {
        assert!(!mesh.dead[mesh.hint], "the hint is a dead triangle");
        assert_ne!(mesh.triangles[mesh.hint][2], GHOST, "the hint is a ghost");

        let mut expected: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (index, triangle) in mesh.triangles.iter().enumerate() {
            if mesh.dead[index] {
                continue;
            }
            for k in 0..3 {
                expected.insert((triangle[k], triangle[(k + 1) % 3]), index);
            }
        }
        assert_eq!(
            mesh.owner, expected,
            "the edge map and the live triangles disagree"
        );
    }

    fn random_points(count: usize, seed: u64, grid: u64) -> Vec<[f64; 2]> {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            let value = ((state >> 33) % grid) as f64;
            value
        };
        (0..count).map(|_| [next(), next()]).collect()
    }

    #[test]
    fn a_square_is_two_triangles_and_its_diagonal_is_a_choice() {
        let square = alloc::vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mesh = Triangulation::new(&square).unwrap();
        assert_eq!(mesh.triangles().len(), 2);
        assert_is_delaunay(&mesh);
        assert_covers_every_point(&mesh);
        // Both diagonals are Delaunay here, and which one comes out is decided by the insertion order
        // rather than by rounding: the last corner lies exactly *on* the seed triangle's circumcircle,
        // so it does not conflict with it and the seed's diagonal survives.
        assert_eq!(mesh.triangles(), [[0, 1, 2], [0, 2, 3]]);
    }

    #[test]
    fn scattered_points_satisfy_the_empty_circumcircle_property() {
        for seed in [1_u64, 2, 3, 4, 5] {
            let points = random_points(60, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), 1_000_000);
            let mesh = Triangulation::new(&points).unwrap();
            assert_is_delaunay(&mesh);
            assert_covers_every_point(&mesh);
        }
    }

    /// The degenerate case that is also the common one: a regular grid. Every cell's four corners are
    /// cocircular, and every boundary row puts points exactly on a convex-hull edge.
    #[test]
    fn a_regular_grid_triangulates_despite_being_cocircular_everywhere() {
        let mut points = Vec::new();
        for i in 0..7 {
            for j in 0..7 {
                points.push([f64::from(i), f64::from(j)]);
            }
        }
        let mesh = Triangulation::new(&points).unwrap();
        assert_is_delaunay(&mesh);
        assert_covers_every_point(&mesh);
        // 36 unit cells, two triangles each.
        assert_eq!(mesh.triangles().len(), 72);
    }

    /// Coarse random coordinates: duplicated points, collinear runs and cocircular quadruples all occur
    /// with probability one rather than never.
    #[test]
    fn a_coarse_lattice_of_points_with_duplicates_still_triangulates() {
        for seed in [11_u64, 12, 13] {
            let points = random_points(120, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), 9);
            let mesh = Triangulation::new(&points).unwrap();
            assert_is_delaunay(&mesh);
            assert_covers_every_point(&mesh);
        }
    }

    /// The triangles tile the convex hull exactly: same area, computed two independent ways.
    #[test]
    fn the_triangles_cover_the_convex_hull_and_nothing_else() {
        let points = random_points(80, 0x5DEE_CE66_D000_0011, 1_000_000);
        let mesh = Triangulation::new(&points).unwrap();

        let mut triangulated = 0.0;
        for t in mesh.triangles() {
            let (a, b, c) = (points[t[0]], points[t[1]], points[t[2]]);
            triangulated += 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]));
        }
        let hull = convex_hull_2d(&points).unwrap();
        let mut shoelace = 0.0;
        for k in 0..hull.len() {
            let (p, q) = (points[hull[k]], points[hull[(k + 1) % hull.len()]]);
            shoelace += p[0] * q[1] - q[0] * p[1];
        }
        let hull_area = 0.5 * shoelace;
        assert!(
            (triangulated - hull_area).abs() < 1e-9 * hull_area,
            "triangulated {triangulated:.6e} vs hull {hull_area:.6e}"
        );
    }

    /// The walk is checked on its own because `locate`'s exhaustive scan would otherwise hide a walk
    /// that lands anywhere at all behind a slower correct answer.
    #[test]
    fn the_walk_lands_on_a_triangle_that_conflicts_with_the_point() {
        for (count, seed, grid) in [(60_usize, 1_u64, 1_000_000_u64), (120, 12, 9)] {
            let points = random_points(count, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), grid);
            let unique = distinct(&points);
            let [a, b, c] = seed_triangle(&points, &unique).unwrap();
            let mut mesh = Mesh::seed(&points, a, b, c);
            assert_internally_consistent(&mesh);

            for &index in &unique {
                if index == a || index == b || index == c {
                    continue;
                }
                let landed = mesh.walk(index);
                assert!(
                    mesh.conflicts(landed, index),
                    "the walk landed on triangle {landed}, whose circumcircle misses point {index}"
                );
                mesh.insert(index);
                assert_internally_consistent(&mesh);
            }
        }
    }

    /// The scan behind the walk, which the walk's own correctness keeps unreachable from
    /// [`Triangulation::new`].
    #[test]
    fn the_exhaustive_scan_finds_a_live_conflicting_triangle() {
        // A triangle and two points outside it on opposite sides: each conflicts with a ghost, with a
        // different one, and with no real triangle.
        let points = alloc::vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-5.0, -5.0], [5.0, 5.0]];
        let mesh = Mesh::seed(&points, 0, 1, 2);

        for point in [3, 4] {
            assert!(
                !mesh.conflicts(0, point),
                "point {point} was meant to lie outside the seed triangle's circumcircle"
            );
            let found = mesh
                .scan(point)
                .expect("a point outside the hull conflicts with a ghost");
            assert!(!mesh.dead[found], "the scan returned a dead triangle");
            assert!(
                mesh.conflicts(found, point),
                "the scan returned triangle {found}, which does not contain point {point}"
            );
        }
        assert_ne!(
            mesh.scan(3),
            mesh.scan(4),
            "the two lie beyond different hull edges"
        );
        // A point that is already a vertex conflicts with nothing, which is how `locate` reports a
        // duplicate: on a circumcircle is not inside it, and an endpoint is not between the endpoints.
        assert_eq!(mesh.scan(0), None);
    }

    /// Strictness at the endpoints, which de-duplication makes unreachable from the public entry point:
    /// a point that coincides with a hull vertex would otherwise open a zero-area cone.
    #[test]
    fn a_point_is_strictly_between_the_endpoints_and_an_endpoint_is_not() {
        let (a, b) = ([0.0, 0.0], [4.0, 2.0]);
        assert!(between(&a, &b, &[2.0, 1.0]));
        assert!(!between(&a, &b, &a));
        assert!(!between(&a, &b, &b));
        assert!(!between(&a, &b, &[6.0, 3.0]));
        assert!(!between(&a, &b, &[-2.0, -1.0]));

        // The vertical branch, where the x comparison is degenerate and the endpoints descend.
        let (c, d) = ([1.0, 5.0], [1.0, 1.0]);
        assert!(between(&c, &d, &[1.0, 3.0]));
        assert!(!between(&c, &d, &c));
        assert!(!between(&c, &d, &d));
        assert!(!between(&c, &d, &[1.0, 7.0]));
    }

    #[test]
    fn a_line_of_points_has_no_triangulation() {
        let line = alloc::vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        assert_eq!(
            Triangulation::new(&line).err(),
            Some(GeometryError::Degenerate)
        );
        assert_eq!(
            Triangulation::new(&[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]).err(),
            Some(GeometryError::TooFewPoints(2))
        );
        assert_eq!(
            Triangulation::new(&[[0.0, 0.0], [1.0, 0.0], [f64::INFINITY, 0.0]]).err(),
            Some(GeometryError::NonFiniteCoordinate { point: 2 })
        );
    }

    /// A triangle with its smallest index first, its cyclic order kept.
    fn smallest_first(t: [usize; 3]) -> [usize; 3] {
        let m = t
            .iter()
            .position(|&v| v == *t.iter().min().unwrap())
            .unwrap();
        [t[m], t[(m + 1) % 3], t[(m + 2) % 3]]
    }

    /// (G2) The triangulation is the Delaunay triangulation by definition: on 14 integer points in
    /// general position — no three collinear, no four cocircular, so the answer is unique —
    /// `derivations/hull_delaunay.mac` tests every counter-clockwise triple against every other point
    /// with the exact lifted determinant and keeps those with no point strictly inside, and
    /// `derivations/hull_delaunay.gp` repeats the search in PARI. The 19 triangles are `2n − 2 − h`
    /// with `h = 7`. The insertion order of Bowyer–Watson must not matter, so the set is compared,
    /// each triangle with its smallest index first.
    #[test]
    fn the_triangulation_is_the_brute_force_delaunay_set() {
        let points = alloc::vec![
            [592.0, 351.0],
            [379.0, 241.0],
            [82.0, 89.0],
            [575.0, 1001.0],
            [247.0, 109.0],
            [1009.0, 661.0],
            [706.0, 585.0],
            [906.0, 503.0],
            [304.0, 539.0],
            [388.0, 863.0],
            [954.0, 397.0],
            [263.0, 368.0],
            [89.0, 482.0],
            [308.0, 897.0]
        ];
        let brute_force: [[usize; 3]; 19] = [
            [0, 1, 4],
            [0, 4, 10],
            [0, 6, 8],
            [0, 7, 6],
            [0, 8, 11],
            [0, 10, 7],
            [0, 11, 1],
            [1, 11, 4],
            [2, 4, 11],
            [2, 11, 12],
            [3, 6, 5],
            [3, 9, 6],
            [3, 13, 9],
            [5, 6, 7],
            [5, 7, 10],
            [6, 9, 8],
            [8, 9, 13],
            [8, 12, 11],
            [8, 13, 12],
        ];
        let mut got: Vec<[usize; 3]> = Triangulation::new(&points)
            .unwrap()
            .triangles()
            .iter()
            .map(|&t| smallest_first(t))
            .collect();
        got.sort_unstable();
        assert_eq!(got, brute_force);
    }
}
