//! Convex hulls in two and three dimensions.
//!
//! Both are built on the exact predicates, so the answer is the hull of the points as given rather
//! than of a perturbation of them: a face is kept only when every other point is behind it *exactly*,
//! and a point is a vertex only when it is strictly extreme. Points that lie exactly on a face — the
//! interior of a coplanar facet, or the interior of a hull edge — are not vertices, which is the same
//! convention qhull reports.
//!
//! Gate: `derivations/hull_delaunay.mac` finds both hulls of a 14-point integer set by brute force
//! from their definitions — every oriented triangle with all other points strictly behind it, every
//! edge with all other points strictly to its left — in exact arithmetic, and checks Euler's
//! `F = 2V − 4`; `derivations/hull_delaunay.gp` repeats the search. The module tests
//! `the_hull_is_the_brute_force_set_of_supporting_planes` (G2) and
//! `the_planar_hull_is_the_brute_force_supporting_cycle` (G2) compare the constructions with it.

use crate::GeometryError;
use crate::predicates::{orient2d, orient3d};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

/// The convex hull of a 2-D point set, as indices into the caller's slice in counter-clockwise order
/// starting from the lexicographically smallest point.
///
/// Andrew's monotone chain: sort, then sweep the lower and upper chains, popping any vertex that does
/// not make a strict left turn. The strictness is what drops points lying exactly on a hull edge.
///
/// # Errors
///
/// [`GeometryError::NonFiniteCoordinate`] for an infinite or NaN coordinate,
/// [`GeometryError::TooFewPoints`] for fewer than three distinct points, and
/// [`GeometryError::Degenerate`] when every point is collinear — a segment is not a polygon, and a
/// caller that wanted the two extremes asked the wrong question.
pub fn convex_hull_2d(points: &[[f64; 2]]) -> Result<Vec<usize>, GeometryError> {
    for (index, p) in points.iter().enumerate() {
        if !p[0].is_finite() || !p[1].is_finite() {
            return Err(GeometryError::NonFiniteCoordinate { point: index });
        }
    }
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&i, &j| {
        points[i][0]
            .total_cmp(&points[j][0])
            .then(points[i][1].total_cmp(&points[j][1]))
    });
    // Exact equality is the intent: only a bit-identical duplicate is a duplicate.
    #[allow(clippy::float_cmp)]
    order.dedup_by(|&mut i, &mut j| points[i] == points[j]);
    if order.len() < 3 {
        return Err(GeometryError::TooFewPoints(order.len()));
    }

    let mut chain: Vec<usize> = Vec::with_capacity(2 * order.len());
    for pass in 0..2 {
        let start = chain.len();
        let sweep: Vec<usize> = if pass == 0 {
            order.clone()
        } else {
            order.iter().rev().copied().collect()
        };
        for &index in &sweep {
            while chain.len() >= start + 2 {
                let (a, b) = (chain[chain.len() - 2], chain[chain.len() - 1]);
                if orient2d(&points[a], &points[b], &points[index]) > 0 {
                    break;
                }
                chain.pop();
            }
            chain.push(index);
        }
        // The last point of each pass is the first of the next.
        chain.pop();
    }
    if chain.len() < 3 {
        return Err(GeometryError::Degenerate);
    }
    Ok(chain)
}

/// The convex hull of a 3-D point set: a closed, outward-oriented triangulated surface.
///
/// A flat facet is reported as a fan of triangles over its own extreme points, so
/// [`faces`](Self::faces) may contain several triangles sharing a supporting plane — how that facet is
/// cut is arbitrary, but which points it is cut from is not. [`vertices`](Self::vertices) holds the
/// extreme points and nothing else, which is what a comparison against another implementation should be
/// made on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvexHull {
    faces: Vec<[usize; 3]>,
    vertices: Vec<usize>,
}

impl ConvexHull {
    /// Build the hull by incremental insertion.
    ///
    /// Each point outside the current hull sees a connected set of faces; those are deleted and the
    /// point is coned to the horizon — the boundary of the deleted region. Visibility is *strict*
    /// (`orient3d < 0`), which is what makes the cone safe: a horizon edge collinear with the new point
    /// would need both its faces to be coplanar with the point, and then neither is visible and the
    /// edge is not on the horizon, so no zero-area face can be created.
    ///
    /// `O(n · f)` predicate evaluations for `n` points and `f` faces — the scan for visible faces is
    /// exhaustive rather than conflict-graph-driven. For the point counts this workspace produces
    /// (station sets, sub-satellite points, patch-sized clouds) that is the right trade; a caller
    /// meshing a million-point cloud should say so and get a different algorithm.
    ///
    /// # Errors
    ///
    /// [`GeometryError::NonFiniteCoordinate`], [`GeometryError::TooFewPoints`] for fewer than four
    /// distinct points, or [`GeometryError::Degenerate`] when every point lies in one plane.
    pub fn new(points: &[[f64; 3]]) -> Result<Self, GeometryError> {
        for (index, p) in points.iter().enumerate() {
            if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
                return Err(GeometryError::NonFiniteCoordinate { point: index });
            }
        }
        let unique = distinct(points);
        if unique.len() < 4 {
            return Err(GeometryError::TooFewPoints(unique.len()));
        }

        let mut faces = initial_tetrahedron(points, &unique)?;
        let seeded: BTreeSet<usize> = faces.iter().flatten().copied().collect();
        for &index in &unique {
            if seeded.contains(&index) {
                continue;
            }
            insert(points, &mut faces, index);
        }
        let faces = retriangulate_flat_facets(points, &faces);

        let vertices: BTreeSet<usize> = faces.iter().flatten().copied().collect();
        Ok(Self {
            faces,
            vertices: vertices.into_iter().collect(),
        })
    }

    /// The hull's triangular faces, each oriented so that its normal `(b − a) × (c − a)` points out of
    /// the hull.
    #[must_use]
    pub fn faces(&self) -> &[[usize; 3]] {
        &self.faces
    }

    /// The indices of the extreme points, ascending.
    #[must_use]
    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }
}

/// Indices of the distinct points, first occurrence winning.
///
/// Exact duplicates are removed here rather than handled downstream: a duplicated point is coplanar
/// with every face that already carries it, and every incremental construction that tries to cone it
/// to a horizon produces a zero-area face. Normalizing once at the boundary is cheaper than a guard in
/// each consumer.
fn distinct(points: &[[f64; 3]]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&i, &j| {
        points[i][0]
            .total_cmp(&points[j][0])
            .then(points[i][1].total_cmp(&points[j][1]))
            .then(points[i][2].total_cmp(&points[j][2]))
    });
    // Exact equality is the intent: only a bit-identical duplicate is a duplicate.
    #[allow(clippy::float_cmp)]
    order.dedup_by(|&mut i, &mut j| points[i] == points[j]);
    order.sort_unstable();
    order
}

/// Four affinely independent points, as an outward-oriented tetrahedron.
fn initial_tetrahedron(
    points: &[[f64; 3]],
    unique: &[usize],
) -> Result<Vec<[usize; 3]>, GeometryError> {
    let a = unique[0];
    let b = unique[1];
    let c = unique[2..]
        .iter()
        .copied()
        .find(|&i| !collinear(&points[a], &points[b], &points[i]))
        .ok_or(GeometryError::Degenerate)?;
    let d = unique
        .iter()
        .copied()
        .find(|&i| orient3d(&points[a], &points[b], &points[c], &points[i]) != 0)
        .ok_or(GeometryError::Degenerate)?;

    // `orient3d > 0` puts `d` behind `(a, b, c)`, which is exactly what an outward face means.
    let (a, b, c) = if orient3d(&points[a], &points[b], &points[c], &points[d]) > 0 {
        (a, b, c)
    } else {
        (a, c, b)
    };
    Ok(alloc::vec![[a, b, c], [b, a, d], [c, b, d], [a, c, d]])
}

/// Exact collinearity in 3-D: the cross product vanishes iff all three of its components do, and each
/// component is a 2-D orientation determinant on a coordinate projection.
fn collinear(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> bool {
    let project = |p: &[f64; 3], first: usize, second: usize| [p[first], p[second]];
    (0..3).all(|axis| {
        let (first, second) = ((axis + 1) % 3, (axis + 2) % 3);
        orient2d(
            &project(a, first, second),
            &project(b, first, second),
            &project(c, first, second),
        ) == 0
    })
}

/// Re-triangulate each maximal coplanar group of faces from its own extreme points.
///
/// The incremental construction cannot do this itself, and the reason is worth stating: a point that
/// is extreme when it is inserted can stop being extreme later, and it stops being extreme *without*
/// every face it carries becoming visible — the ones in the plane it now lies in are coplanar with the
/// new point, not in front of it, so they survive and keep it as a vertex. On a regular grid that is
/// not an edge case: inserting `(0,0,2)` after `(0,0,0)` and `(0,0,1)` leaves the middle point on a
/// hull edge, carrying slivers. qhull's answer for the 3×3×3 grid is 8 vertices; without this pass
/// ours is 26, and the surface is a correct hull in both cases — the disagreement is entirely about
/// what counts as a vertex.
///
/// A facet of a convex body is the intersection with a supporting plane, so it is convex and connected:
/// its extreme points are the 2-D hull of the group's vertices in the facet's own plane, and a fan from
/// one of them covers it. Both the grouping (exact coplanarity, against the seed face's plane so the
/// test cannot drift across a chain of near-coplanar neighbours) and the 2-D hull are exact.
fn retriangulate_flat_facets(points: &[[f64; 3]], faces: &[[usize; 3]]) -> Vec<[usize; 3]> {
    let mut owner: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (index, face) in faces.iter().enumerate() {
        for k in 0..3 {
            owner.insert((face[k], face[(k + 1) % 3]), index);
        }
    }

    let mut group_of = alloc::vec![usize::MAX; faces.len()];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for start in 0..faces.len() {
        if group_of[start] != usize::MAX {
            continue;
        }
        let id = groups.len();
        group_of[start] = id;
        let (mut stack, mut group) = (alloc::vec![start], Vec::new());
        let seed = faces[start];
        while let Some(index) = stack.pop() {
            group.push(index);
            let face = faces[index];
            for k in 0..3 {
                let across = owner[&(face[(k + 1) % 3], face[k])];
                if group_of[across] != usize::MAX {
                    continue;
                }
                let coplanar = faces[across].iter().all(|&v| {
                    orient3d(
                        &points[seed[0]],
                        &points[seed[1]],
                        &points[seed[2]],
                        &points[v],
                    ) == 0
                });
                if coplanar {
                    group_of[across] = id;
                    stack.push(across);
                }
            }
        }
        groups.push(group);
    }

    let mut out = Vec::with_capacity(faces.len());
    for group in &groups {
        if group.len() == 1 {
            out.push(faces[group[0]]);
            continue;
        }
        out.extend(fan_over_facet(points, faces, group));
    }
    out
}

/// `(b − a) × (c − a)`, in floating point.
///
/// Only which component is largest is ever read, to pick a projection axis, so this deliberately does
/// not go through the exact predicates: the hull computed in the chosen projection is exact whichever
/// axis it is, and a rounded cross product is enough to choose one that does not collapse.
fn plane_normal(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> [f64; 3] {
    let edge1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let edge2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    [
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    ]
}

/// One coplanar group as a fan over its extreme points, oriented outward like the group it replaces.
fn fan_over_facet(points: &[[f64; 3]], faces: &[[usize; 3]], group: &[usize]) -> Vec<[usize; 3]> {
    let seed = faces[group[0]];
    let (a, b, c) = (&points[seed[0]], &points[seed[1]], &points[seed[2]]);
    let normal = plane_normal(a, b, c);
    // Project along the axis the facet is most nearly perpendicular to, which is the one where the
    // projection cannot collapse. Only the choice of axis uses floating point; the hull in it does not.
    let dropped = (0..3)
        .max_by(|&i, &j| libm::fabs(normal[i]).total_cmp(&libm::fabs(normal[j])))
        .unwrap_or(0);
    let (first, second) = ((dropped + 1) % 3, (dropped + 2) % 3);

    let members: Vec<usize> = group
        .iter()
        .flat_map(|&index| faces[index])
        .collect::<BTreeSet<usize>>()
        .into_iter()
        .collect();
    let projected: Vec<[f64; 2]> = members
        .iter()
        .map(|&v| [points[v][first], points[v][second]])
        .collect();
    let Ok(ring) = convex_hull_2d(&projected) else {
        // A facet whose projection is degenerate would be a zero-area facet, which the construction
        // cannot produce; keeping the group unchanged is the honest fallback if one ever appears.
        return group.iter().map(|&index| faces[index]).collect();
    };
    let mut ring: Vec<usize> = ring.into_iter().map(|k| members[k]).collect();

    // The projection's own orientation is not the facet's: a witness strictly behind the plane says
    // which way round the fan has to run.
    if let Some(witness) = points.iter().find(|p| orient3d(a, b, c, p) > 0)
        && orient3d(
            &points[ring[0]],
            &points[ring[1]],
            &points[ring[2]],
            witness,
        ) < 0
    {
        ring.reverse();
    }

    (1..ring.len() - 1)
        .map(|k| [ring[0], ring[k], ring[k + 1]])
        .collect()
}

/// Cone one point to the horizon of the faces it can see.
fn insert(points: &[[f64; 3]], faces: &mut Vec<[usize; 3]>, index: usize) {
    let p = &points[index];
    let mut visible = Vec::new();
    for (position, face) in faces.iter().enumerate() {
        if orient3d(&points[face[0]], &points[face[1]], &points[face[2]], p) < 0 {
            visible.push(position);
        }
    }
    if visible.is_empty() {
        return;
    }

    let mut edges = BTreeSet::new();
    for &position in &visible {
        let face = faces[position];
        for k in 0..3 {
            edges.insert((face[k], face[(k + 1) % 3]));
        }
    }
    let horizon: Vec<(usize, usize)> = edges
        .iter()
        .copied()
        .filter(|&(u, v)| !edges.contains(&(v, u)))
        .collect();

    for &position in visible.iter().rev() {
        faces.swap_remove(position);
    }
    for (u, v) in horizon {
        faces.push([u, v, index]);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]
    use super::*;

    /// Every input point behind every face, and every face's own vertices on it. This is the defining
    /// property of a hull and it is checked directly rather than against a golden file.
    fn assert_is_a_hull(points: &[[f64; 3]], hull: &ConvexHull) {
        for face in hull.faces() {
            let (a, b, c) = (&points[face[0]], &points[face[1]], &points[face[2]]);
            for (index, p) in points.iter().enumerate() {
                assert!(
                    orient3d(a, b, c, p) >= 0,
                    "point {index} is outside face {face:?}"
                );
            }
        }
        // A closed oriented surface: every directed edge appears exactly once, and its reverse too.
        let mut edges = BTreeSet::new();
        for face in hull.faces() {
            for k in 0..3 {
                assert!(
                    edges.insert((face[k], face[(k + 1) % 3])),
                    "directed edge repeated in {face:?}"
                );
            }
        }
        for &(u, v) in &edges {
            assert!(edges.contains(&(v, u)), "edge {u}->{v} has no twin");
        }
        // Euler's formula for a closed triangulated surface: E = 3F/2, so V − E + F = 2 is V = F/2 + 2.
        assert_eq!(
            2 * hull.vertices().len(),
            hull.faces().len() + 4,
            "V - E + F != 2"
        );
    }

    #[test]
    fn the_hull_of_a_cube_is_its_eight_corners_and_twelve_triangles() {
        let mut points = Vec::new();
        for x in [0.0, 1.0] {
            for y in [0.0, 1.0] {
                for z in [0.0, 1.0] {
                    points.push([x, y, z]);
                }
            }
        }
        // A point in the middle of the cube and one in the middle of a face: neither is extreme.
        points.push([0.5, 0.5, 0.5]);
        points.push([0.5, 0.5, 0.0]);

        let hull = ConvexHull::new(&points).unwrap();
        assert_eq!(hull.vertices(), &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(hull.faces().len(), 12, "8 vertices means 2·8 − 4 triangles");
        assert_is_a_hull(&points, &hull);
    }

    #[test]
    fn duplicated_and_coplanar_points_do_not_break_the_construction() {
        let mut points = alloc::vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        points.extend_from_slice(&points.clone()); // every vertex duplicated
        points.push([0.25, 0.25, 0.0]); // inside a face
        points.push([0.5, 0.5, 0.0]); // on a face's edge

        let hull = ConvexHull::new(&points).unwrap();
        assert_eq!(hull.faces().len(), 4);
        assert_eq!(hull.vertices(), &[0, 1, 2, 3]);
        assert_is_a_hull(&points, &hull);
    }

    #[test]
    fn a_flat_point_set_has_no_three_dimensional_hull() {
        let flat = alloc::vec![
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0, 3.0, 1.0],
        ];
        assert_eq!(
            ConvexHull::new(&flat).err(),
            Some(GeometryError::Degenerate)
        );

        let collinear = alloc::vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ];
        assert_eq!(
            ConvexHull::new(&collinear).err(),
            Some(GeometryError::Degenerate)
        );
    }

    #[test]
    fn the_two_dimensional_hull_drops_points_on_its_own_edges() {
        let points = alloc::vec![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 0.0], // on the bottom edge
            [1.0, 1.0], // interior
        ];
        let hull = convex_hull_2d(&points).unwrap();
        assert_eq!(hull.len(), 4);
        for &index in &hull {
            assert!(index < 4, "a non-extreme point entered the hull: {index}");
        }
        // Counter-clockwise: every consecutive triple turns left.
        for k in 0..hull.len() {
            let (a, b, c) = (
                hull[k],
                hull[(k + 1) % hull.len()],
                hull[(k + 2) % hull.len()],
            );
            assert_eq!(orient2d(&points[a], &points[b], &points[c]), 1);
        }
    }

    /// Three points are a polygon, and the smallest one. Both length guards sit one point away from it.
    #[test]
    fn a_triangle_is_a_hull_and_starts_at_its_smallest_point() {
        let points = alloc::vec![[2.0, 1.0], [0.0, 3.0], [1.0, 0.0], [2.0, 1.0]];
        let hull = convex_hull_2d(&points).unwrap();
        assert_eq!(hull.len(), 3, "three distinct points are a triangle");
        // Counter-clockwise from the lexicographically smallest point, which is the documented order and
        // not merely the order the two sweeps happen to produce.
        assert_eq!(hull, alloc::vec![1, 2, 0]);
    }

    /// Three edges whose cross product has no zero, no repeated and no equal-magnitude component, so
    /// that any operator change in it moves the largest component and with it the projection axis.
    #[test]
    fn the_facet_normal_is_the_cross_product_of_its_two_edges() {
        let normal = plane_normal(&[1.0, 2.0, 3.0], &[3.0, 7.0, 13.0], &[5.0, 11.0, 29.0]);
        // (b − a) = (2, 5, 10) and (c − a) = (4, 9, 26), so the cross product is
        // (5·26 − 10·9, 10·4 − 2·26, 2·9 − 5·4).
        assert_eq!(normal, [40.0, -12.0, -2.0]);
    }

    /// Each of these three is collinear in one coordinate projection and not in the other two, so a
    /// projection dropped from the test would report them collinear in space.
    #[test]
    fn collinearity_in_one_projection_is_not_collinearity_in_space() {
        assert!(!collinear(
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0]
        ));
        assert!(!collinear(
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 0.0, 2.0]
        ));
        assert!(!collinear(
            &[0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0]
        ));
        assert!(collinear(
            &[0.0, 0.0, 0.0],
            &[1.0, 2.0, 3.0],
            &[2.0, 4.0, 6.0]
        ));
    }

    #[test]
    fn a_line_of_points_has_no_polygon() {
        let line = alloc::vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        assert_eq!(convex_hull_2d(&line).err(), Some(GeometryError::Degenerate));
        assert_eq!(
            convex_hull_2d(&[[0.0, 0.0], [1.0, 1.0]]).err(),
            Some(GeometryError::TooFewPoints(2))
        );
        assert_eq!(
            convex_hull_2d(&[[0.0, f64::NAN]]).err(),
            Some(GeometryError::NonFiniteCoordinate { point: 0 })
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

    /// (G2) The hull is the set of supporting planes by definition: on 14 integer points with no four
    /// coplanar, `derivations/hull_delaunay.mac` keeps every oriented triangle with all other points
    /// strictly behind it (`orient3d > 0` in exact integers), and `derivations/hull_delaunay.gp`
    /// repeats the search in PARI. Its 16 faces on 10 vertices are `F = 2V − 4`.
    #[test]
    fn the_hull_is_the_brute_force_set_of_supporting_planes() {
        let points = alloc::vec![
            [-142.0, -184.0, 433.0],
            [-888.0, 345.0, 702.0],
            [-71.0, -467.0, 244.0],
            [-1001.0, 358.0, 189.0],
            [-401.0, -503.0, -953.0],
            [-1005.0, -285.0, 285.0],
            [70.0, -199.0, 94.0],
            [-264.0, -284.0, -949.0],
            [393.0, 387.0, 94.0],
            [-143.0, -710.0, 786.0],
            [212.0, -882.0, -1003.0],
            [-23.0, 742.0, -485.0],
            [-624.0, -76.0, 498.0],
            [-945.0, 502.0, -707.0]
        ];
        let brute_force: [[usize; 3]; 16] = [
            [1, 3, 5],
            [1, 5, 9],
            [1, 8, 11],
            [1, 9, 8],
            [1, 11, 13],
            [1, 13, 3],
            [3, 13, 5],
            [4, 5, 13],
            [4, 7, 10],
            [4, 10, 5],
            [4, 13, 7],
            [5, 10, 9],
            [7, 11, 10],
            [7, 13, 11],
            [8, 9, 10],
            [8, 10, 11],
        ];
        let hull = ConvexHull::new(&points).unwrap();
        let mut got: Vec<[usize; 3]> = hull.faces().iter().map(|&t| smallest_first(t)).collect();
        got.sort_unstable();
        assert_eq!(got, brute_force);
        assert_eq!(hull.vertices(), &[1, 3, 4, 5, 7, 8, 9, 10, 11, 13]);
    }

    /// (G2) The planar hull is the cycle of edges with every other point strictly to their left,
    /// found by brute force in `derivations/hull_delaunay.mac` and `derivations/hull_delaunay.gp` on
    /// the 14 planar points of the Delaunay gate, listed counter-clockwise from the lexicographically
    /// smallest point.
    #[test]
    fn the_planar_hull_is_the_brute_force_supporting_cycle() {
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
        assert_eq!(convex_hull_2d(&points).unwrap(), [2, 4, 10, 5, 3, 13, 12]);
    }
}
