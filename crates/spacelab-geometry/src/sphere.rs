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
//!
//! Gate: `derivations/spherical_triangle.mac` proves the Van Oosterom–Strackee half-angle formula is
//! Girard's excess (and checks it against Girard and l'Huilier at 60 digits), the exact cell areas of
//! the tetrahedron and the octahedron, and that the face normal is the circumcentre; it prints 80-digit
//! excesses of 16 double-valued triangles that `derivations/spherical_triangle.gp` re-derives by
//! l'Huilier. The module tests `the_solid_angle_of_a_spherical_triangle_is_its_excess` (G1),
//! `a_tetrahedron_of_directions_is_four_triangles_and_four_equal_cells` (G3),
//! `the_cells_of_an_octahedron_are_the_six_faces_of_a_cube` (G3) and
//! `the_cells_tile_the_sphere_and_their_corners_are_circumcentres` (G4) tie the code to it.

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
///
/// That holds only if the triple product is formed from the edges. `a·(b × c)` of three unit vectors
/// a side `ℓ` apart takes the dot product of `a` with a cross product of length `ℓ` to get a volume of
/// order `ℓ²`, so the `ε` the cross product carries becomes a relative error `ε/ℓ²` — 9.7e-5 at
/// `ℓ = 1e-6` rad, as bad as Girard's sum. `a·((b − a) × (c − a))` is the same number
/// (`derivations/spherical_triangle.mac`), but the differences of nearby components are exact and
/// their cross product is already the small volume, so nothing cancels: 2.7e-15 at worst on the same
/// triangles.
fn spherical_excess(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let triple = dot(a, &cross(&ab, &ac));
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

    /// (G3) A regular tetrahedron's cell is a quarter of the sphere, `π` exactly:
    /// `derivations/spherical_triangle.mac` shows each of its three fan triangles has
    /// `tan(Ω/2) = 1/√3`, so `Ω = π/3`.
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

    /// (G3) An octahedron's cell is a sixth of the sphere, `2π/3`: `derivations/spherical_triangle.mac`
    /// shows each of its four fan triangles has `tan(Ω/2) = 2 − √3 = tan(π/12)`.
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

    /// (G4) The two properties the ticket asks for, on a diagram with no symmetry: the cells tile the
    /// sphere, and each Voronoi vertex is equidistant from the three generators whose triangle it
    /// duals — the face normal `(b − a) × (c − a)` that `derivations/spherical_triangle.mac` shows has
    /// `n·a = n·b = n·c`.
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

    /// (G1) The Van Oosterom–Strackee formula is the spherical excess. `derivations/spherical_triangle.mac`
    /// proves `D² + T² = 2(1 + a·b)(1 + b·c)(1 + c·a)`, which makes `T` and `D` the sine and cosine of
    /// half of Girard's excess, and evaluates these 16 triangles at 80 digits on the exactly normalized
    /// doubles; `derivations/spherical_triangle.gp` re-evaluates them by l'Huilier's theorem and agrees
    /// to 9e-70 relative. Sides run from about 1 rad down to 1e-6 rad and both orientations occur, so
    /// the sign is gated too. The vertices are the nearest doubles to unit vectors, as the module's
    /// own normalization produces them.
    ///
    /// The bound is a few ulps at every size, which is the property the edge-difference triple
    /// product buys: measured 2.7e-15 at worst (sides of 2.4 rad) and at most 2.1e-16 below 1e-2 rad.
    /// The triple product `a·(b × c)` it replaced was out by `ε/ℓ²` — 9.7e-5 at `ℓ = 1.6e-6` rad,
    /// 3.1e-9 at 1e-4 rad.
    #[test]
    #[allow(clippy::excessive_precision, clippy::unreadable_literal)]
    fn the_solid_angle_of_a_spherical_triangle_is_its_excess() {
        #[rustfmt::skip]
        let rows: [(&[(i64, i32)], f64); 16] = [
            (&[(-1845248265345387, -52), (-8101874579790437, -53), (2734594715793605, -54), (-8762861518068887, -54), (-5595792367931605, -55), (3872197281722967, -52), (-6825180652953691, -55), (-5844451856231675, -53), (3318907114198361, -52)], -2.04019640851864893666724057149e-1),
            (&[(7625040251960805, -57), (6980042243268611, -53), (5672877090379971, -53), (532629342956981, -49), (-1958846551995637, -54), (5493523501168025, -54), (-3171774373307499, -53), (-5991308764029777, -55), (-2074035912257115, -51)], 3.81044254238013958451814098398e+0),
            (&[(344287104409255, -49), (-6736986770453353, -53), (-290421123170217, -50), (-7297724370529969, -53), (8676812698472637, -56), (5166865280146781, -53), (-2372875986074091, -52), (-2649104933553353, -52), (1381494861498575, -51)], -7.16652453023707117603107611488e-2),
            (&[(-5401579990806579, -56), (545198090995785, -49), (-1070053899955221, -52), (-348845823985691, -50), (5642837203820341, -53), (6442022084361661, -53), (-2941418897860469, -52), (8014278690130951, -54), (2759741480332539, -52)], -2.47774865626423334484658020888e-1),
            (&[(-2837640678995401, -52), (4408751695311781, -57), (-3494456101581985, -52), (-5672956061980739, -53), (6397294314953431, -58), (-437085704496969, -49), (-2842892192441313, -52), (6577198877384375, -58), (-6982782303190223, -53)], 7.59592494824739255409345779199e-6),
            (&[(-7326125788269101, -53), (2618028431619009, -52), (-812185929483295, -55), (-1829250326017861, -51), (2624792403591585, -52), (-1453044979677951, -56), (-3666656035836497, -52), (2613424115920355, -52), (-718109020864801, -55)], -3.90551197009037095746379458580e-6),
            (&[(6957339230063655, -53), (-8877395254254193, -54), (-3608743912687659, -53), (1724655934887889, -51), (-2241211270472727, -52), (-7333898575553519, -54), (1735072314662255, -51), (-4525705598706389, -53), (-883176520249681, -51)], -6.72869612836562049727074673566e-5),
            (&[(-7708186806451501, -54), (-5857779288186073, -53), (1413374160137645, -51), (-3841515496964161, -53), (-1469811161110833, -51), (5639758816826769, -53), (-7629011685814805, -54), (-1474626405015133, -51), (1409494923703751, -51)], 3.30019282324538675476738732003e-6),
            (&[(-4627419503233571, -56), (-668863388690317, -50), (-7222385562883461, -53), (-4623804075009657, -56), (-5349633272119435, -53), (-7223365322271933, -53), (-4613319799819321, -56), (-5350851578242849, -53), (-3611283819808995, -52)], 1.70618596028759591505571234291e-8),
            (&[(7696366429872485, -54), (-7966273560800747, -53), (-6764152468271691, -55), (481116395097395, -50), (-995729277085789, -50), (-6765622001216959, -55), (3848422344753369, -53), (-7966200584758605, -53), (-6763350784220883, -55)], 1.65679608586405804784566337802e-9),
            (&[(8105010309963599, -53), (3865916405111903, -53), (1404473764336103, -54), (8105585363476163, -53), (3864845105645185, -53), (5611968007767069, -56), (8105528325195381, -53), (1932560733324273, -52), (2802527522404405, -55)], 7.74249149636897331299709120570e-9),
            (&[(4656224924547903, -53), (8595215848313327, -54), (-6401544657100457, -53), (2327997485829779, -52), (2148595234387625, -52), (-100031127130063, -47), (4656324242165371, -53), (8593337479705565, -54), (-6402102834725609, -53)], -2.23225212373914872705974617785e-9),
            (&[(-7946082095610031, -55), (-8794206898992453, -58), (-8781107513228095, -53), (-7946050167911121, -55), (-4396954743527129, -57), (-2195277402454499, -51), (-7946022914112851, -55), (-2198490597284469, -56), (-4390555549727437, -52)], 4.83737374377033254705764270757e-13),
            (&[(4593268218137639, -54), (-2787149068908227, -52), (1672989660278949, -51), (4593256615738273, -54), (-2787151448544935, -52), (6691956667641857, -53), (287078206119881, -50), (-2787150157444345, -52), (418247483193367, -49)], -2.29232471523500644660528083022e-13),
            (&[(-1298862788186317, -52), (7172911893376783, -54), (-1960847934997281, -51), (-5195448647594425, -54), (3586455088167337, -53), (-7843392547406869, -53), (-2597736152905803, -53), (7172893420628471, -54), (-1960848115111307, -51)], 1.46145130339642458804159794892e-13),
            (&[(4613692586931799, -53), (-712768318581513, -50), (5227714975550231, -53), (4613693803523863, -53), (-5702152510987629, -53), (5227707398407993, -53), (576712153338293, -50), (-1425536735525637, -51), (163365951611877, -48)], 2.88669940058302769281442016245e-13),
        ];
        #[allow(clippy::cast_precision_loss)]
        let double = |(mantissa, exponent): (i64, i32)| libm::ldexp(mantissa as f64, exponent);
        for (row, &(c, want)) in rows.iter().enumerate() {
            let v = |k: usize| [double(c[3 * k]), double(c[3 * k + 1]), double(c[3 * k + 2])];
            let (a, b, cc) = (v(0), v(1), v(2));
            let got = spherical_excess(&a, &b, &cc);
            let side = libm::acos(dot(&a, &b).min(1.0)).max(libm::acos(dot(&a, &cc).min(1.0)));
            let relative = libm::fabs(got - want) / libm::fabs(want);
            println!("row {row}: side {side:.1e}, relative {relative:.2e}");
            assert!(relative < 1e-14, "row {row}: {got:e} against {want:e}");
        }
    }
}
