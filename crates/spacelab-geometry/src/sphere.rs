//! Spherical Delaunay triangulation and its dual, the spherical Voronoi diagram.
//!
//! Both come out of one construction: **the convex hull of points on a sphere is their Delaunay
//! triangulation**. A hull face `(a, b, c)` is a face exactly when no other point is in front of its
//! plane, and the plane through `a`, `b`, `c` cuts the sphere in the circle through them — so "no point
//! in front of the plane" and "no point inside the circumcircle" are the same statement. Nothing else
//! is needed: no lifting map, no separate spherical predicate, and the exactness of [`crate::predicates`]
//! carries over unchanged.
//!
//! The dual follows from the same face: the point equidistant from `a`, `b` and `c` on the sphere is the
//! unit normal of that plane, which is the Voronoi vertex. Walking the faces around one generator in
//! order gives its cell.
//!
//! The intended use is the coverage question a grid cannot answer. H3 cells count *whether* a point is
//! covered; the Voronoi diagram of the sub-satellite points says *which* satellite is nearest, and the
//! cell's own geometry says how far the worst-served point in it is from the nearest satellite.

use crate::GeometryError;
use crate::hull::ConvexHull;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// One generator's Voronoi cell on the unit sphere.
#[derive(Debug, Clone, PartialEq)]
pub struct VoronoiCell {
    /// Index of the generating direction in the caller's slice.
    pub generator: usize,
    /// The cell's corners, in order around the generator: the circumcentres of the Delaunay triangles
    /// that meet at it, each a unit vector.
    pub vertices: Vec<[f64; 3]>,
    /// Solid angle of the cell, in steradians. Over a whole diagram these sum to `4π`.
    pub area: f64,
}

/// The Delaunay triangulation of directions on the unit sphere, as triangles of indices into the
/// caller's slice, each oriented counter-clockwise seen from outside the sphere.
///
/// The directions need not be normalized; only their directions matter. A direction that is a duplicate
/// of another — after normalization, exactly — is not a vertex of any triangle, since it generates no
/// cell of its own.
///
/// # Errors
///
/// [`GeometryError::NonFiniteCoordinate`] or [`GeometryError::ZeroLengthDirection`] for an input that is
/// not a direction, [`GeometryError::TooFewPoints`] for fewer than four distinct ones, and
/// [`GeometryError::Degenerate`] when they all lie on one great circle — a triangulation of a circle
/// has no interior.
pub fn spherical_delaunay(directions: &[[f64; 3]]) -> Result<Vec<[usize; 3]>, GeometryError> {
    let unit = normalize_all(directions)?;
    Ok(ConvexHull::new(&unit)?.faces().to_vec())
}

/// The spherical Voronoi diagram: one cell per generator, in ascending index order.
///
/// # Errors
///
/// As [`spherical_delaunay`].
pub fn spherical_voronoi(directions: &[[f64; 3]]) -> Result<Vec<VoronoiCell>, GeometryError> {
    let unit = normalize_all(directions)?;
    let hull = ConvexHull::new(&unit)?;
    let faces = hull.faces();

    let circumcentre: Vec<[f64; 3]> = faces
        .iter()
        .map(|f| unit_normal(&unit[f[0]], &unit[f[1]], &unit[f[2]]))
        .collect();
    // Directed edge -> the face that carries it, which is how the fan around a generator is walked.
    let mut carrier: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (index, face) in faces.iter().enumerate() {
        for k in 0..3 {
            carrier.insert((face[k], face[(k + 1) % 3]), index);
        }
    }

    let mut cells = Vec::with_capacity(hull.vertices().len());
    for &generator in hull.vertices() {
        let first = faces
            .iter()
            .position(|f| f.contains(&generator))
            .ok_or(GeometryError::Degenerate)?;
        let mut vertices = Vec::new();
        let mut face = first;
        loop {
            vertices.push(circumcentre[face]);
            // The generator's outgoing edge in this face, then the face on its other side.
            let f = faces[face];
            let slot = f.iter().position(|&v| v == generator).unwrap_or(0);
            let next_vertex = f[(slot + 2) % 3];
            face = carrier[&(generator, next_vertex)];
            if face == first {
                break;
            }
        }
        let area = cell_area(&unit[generator], &vertices);
        cells.push(VoronoiCell {
            generator,
            vertices,
            area,
        });
    }
    Ok(cells)
}

/// The solid angle of a cell, as the sum of the signed spherical triangles from its generator.
///
/// A Voronoi cell is convex and contains its generator, so the fan is a partition of it whatever the
/// cell's size — including cells larger than a hemisphere, where a formula built on the polygon's
/// interior angles alone has to decide which side is the inside. The sign of the sum is the fan's
/// orientation, which is the same for every cell of one diagram; the magnitude is the area.
fn cell_area(generator: &[f64; 3], vertices: &[[f64; 3]]) -> f64 {
    let mut total = 0.0;
    for k in 0..vertices.len() {
        total += spherical_excess(generator, &vertices[k], &vertices[(k + 1) % vertices.len()]);
    }
    libm::fabs(total)
}

/// The signed area of the spherical triangle `a b c`, by Van Oosterom & Strackee's tangent formula
/// (*The solid angle of a plane triangle*, IEEE TBME 30(2), 1983): the half-angle form is well
/// conditioned for small triangles, where summing interior angles and subtracting `(n − 2)π` is not.
fn spherical_excess(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    let triple = dot(a, &cross(b, c));
    let denominator = 1.0 + dot(a, b) + dot(b, c) + dot(c, a);
    2.0 * libm::atan2(triple, denominator)
}

/// The unit normal of the plane through three points, oriented by `(b − a) × (c − a)` — for an
/// outward-oriented hull face on the unit sphere, the circumcentre of the spherical triangle.
fn unit_normal(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> [f64; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = cross(&ab, &ac);
    let norm = libm::sqrt(dot(&n, &n));
    [n[0] / norm, n[1] / norm, n[2] / norm]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Every direction as a unit vector, refusing the ones that are not directions.
fn normalize_all(directions: &[[f64; 3]]) -> Result<Vec<[f64; 3]>, GeometryError> {
    directions
        .iter()
        .enumerate()
        .map(|(index, d)| {
            if !d[0].is_finite() || !d[1].is_finite() || !d[2].is_finite() {
                return Err(GeometryError::NonFiniteCoordinate { point: index });
            }
            let norm = libm::sqrt(dot(d, d));
            if norm == 0.0 {
                return Err(GeometryError::ZeroLengthDirection { point: index });
            }
            Ok([d[0] / norm, d[1] / norm, d[2] / norm])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]
    use super::*;
    use core::f64::consts::PI;

    /// Points spread by the golden-angle spiral: no symmetry for a bug to hide behind, and no two
    /// points close enough for the normalization to collapse them.
    fn fibonacci_sphere(count: usize) -> Vec<[f64; 3]> {
        let golden = PI * (3.0 - libm::sqrt(5.0));
        (0..count)
            .map(|k| {
                #[allow(clippy::cast_precision_loss)]
                let (index, total) = (k as f64, count as f64);
                let z = 1.0 - 2.0 * (index + 0.5) / total;
                let radius = libm::sqrt(1.0 - z * z);
                let theta = golden * index;
                [radius * libm::cos(theta), radius * libm::sin(theta), z]
            })
            .collect()
    }

    #[test]
    fn a_tetrahedron_of_directions_is_four_triangles_and_four_equal_cells() {
        let directions = alloc::vec![
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ];
        assert_eq!(spherical_delaunay(&directions).unwrap().len(), 4);

        let cells = spherical_voronoi(&directions).unwrap();
        assert_eq!(cells.len(), 4);
        for cell in &cells {
            assert_eq!(cell.vertices.len(), 3);
            assert!(
                (cell.area - PI).abs() < 1e-12,
                "a quarter of the sphere is π steradians, got {}",
                cell.area
            );
        }
    }

    #[test]
    fn the_cells_of_an_octahedron_are_the_six_faces_of_a_cube() {
        let directions = alloc::vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        assert_eq!(spherical_delaunay(&directions).unwrap().len(), 8);
        let cells = spherical_voronoi(&directions).unwrap();
        assert_eq!(cells.len(), 6);
        for cell in &cells {
            assert_eq!(cell.vertices.len(), 4, "each cell is a spherical square");
            assert!((cell.area - 4.0 * PI / 6.0).abs() < 1e-12);
        }
    }

    /// The two properties the ticket asks for, on a diagram with no symmetry: the cells tile the sphere,
    /// and each Voronoi vertex is equidistant from the three generators whose triangle it duals.
    #[test]
    fn the_cells_tile_the_sphere_and_their_corners_are_circumcentres() {
        let directions = fibonacci_sphere(200);
        let triangles = spherical_delaunay(&directions).unwrap();
        let cells = spherical_voronoi(&directions).unwrap();
        assert_eq!(cells.len(), 200);
        assert_eq!(triangles.len(), 2 * 200 - 4);

        let total: f64 = cells.iter().map(|c| c.area).sum();
        assert!(
            (total - 4.0 * PI).abs() < 1e-9,
            "the cells must tile the sphere: {total:.12} vs {:.12}",
            4.0 * PI
        );

        let unit = normalize_all(&directions).unwrap();
        let mut worst_spread = 0.0_f64;
        let mut worst_margin = 0.0_f64;
        for triangle in &triangles {
            let centre = unit_normal(&unit[triangle[0]], &unit[triangle[1]], &unit[triangle[2]]);
            let radius = dot(&centre, &unit[triangle[0]]);
            for &vertex in triangle {
                worst_spread = worst_spread.max((dot(&centre, &unit[vertex]) - radius).abs());
            }
            // The spherical Delaunay property: no other generator is inside the circumcap.
            for (index, p) in unit.iter().enumerate() {
                if !triangle.contains(&index) {
                    worst_margin = worst_margin.max(dot(&centre, p) - radius);
                }
            }
        }
        println!(
            "circumcentre spread {worst_spread:.3e}, worst intruder margin {worst_margin:.3e}"
        );
        assert!(worst_spread < 1e-14, "not equidistant: {worst_spread:.3e}");
        assert!(
            worst_margin < 1e-14,
            "a generator is inside a circumcap: {worst_margin:.3e}"
        );
    }

    #[test]
    fn directions_on_one_great_circle_have_no_diagram() {
        let equator: Vec<[f64; 3]> = (0..8)
            .map(|k| {
                let angle = f64::from(k) * PI / 4.0;
                [libm::cos(angle), libm::sin(angle), 0.0]
            })
            .collect();
        assert_eq!(
            spherical_delaunay(&equator).err(),
            Some(GeometryError::Degenerate)
        );
        assert_eq!(
            spherical_voronoi(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).err(),
            Some(GeometryError::ZeroLengthDirection { point: 0 })
        );
    }

    /// One bad component is enough, and the refusal has to happen here: a NaN survives normalization
    /// as a NaN in every component, so the hull's own finiteness check would report the same error from
    /// further away, while an infinity normalizes to a *finite* direction and would be silently kept.
    #[test]
    fn a_direction_with_one_non_finite_component_is_not_a_direction() {
        for slot in 0..3 {
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut direction = [1.0, 2.0, 3.0];
                direction[slot] = bad;
                assert_eq!(
                    normalize_all(&[[0.0, 0.0, 1.0], direction]).err(),
                    Some(GeometryError::NonFiniteCoordinate { point: 1 }),
                    "component {slot} = {bad}"
                );
            }
        }
    }
}
