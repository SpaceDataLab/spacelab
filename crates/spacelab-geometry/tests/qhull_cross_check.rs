//! `spacelab-geometry` against qhull on the same points: a comparator. The constructions are decided
//! by `derivations/hull_delaunay.mac` — the brute-force Delaunay and hull sets (G2) and Euler's counts
//! (G4) that `delaunay.rs` and `hull.rs` hold their tests to.
//!
//! qhull (Barber, Dobkin & Huhdanpaa, *The quickhull algorithm for convex hulls*, ACM TOMS 22(4),
//! 1996) is the reference implementation, and it reaches its answers by a different route: floating
//! point with facet merging and a joggle fallback, where this crate decides every sign exactly. Two
//! codes with nothing in common agreeing on the combinatorics is a verification; the crate's own
//! tests check the defining predicates, which is a different question.
//!
//! The fixture is written by `tools/gen_qhull_oracle.py` — qhull is an oracle here, never a
//! dependency. Coordinates are stored with 17 significant digits, so both codes see bit-identical
//! doubles.
//!
//! **What is compared depends on whether the input has one answer.** In general position the Delaunay
//! triangulation and the hull's facets are unique and must match exactly. On a regular grid every cell
//! is cocircular and both codes produce *a* valid triangulation; there the comparison is on the
//! invariants — the triangle count, which Euler's formula fixes for any valid triangulation, and the
//! extreme points. Comparing a triangle list in that case would be measuring insertion order.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use spacelab_geometry::hull::ConvexHull;
use spacelab_geometry::{Triangulation, convex_hull_2d};
use std::collections::BTreeSet;

const FIXTURE: &str = include_str!("fixtures/qhull_oracle.txt");

/// One case as the generator wrote it.
struct Case {
    name: String,
    kind: String,
    dimension: usize,
    points: Vec<Vec<f64>>,
    vertices: Vec<usize>,
    simplices: Vec<[usize; 3]>,
    simplex_count: Option<usize>,
}

fn parse(text: &str) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .peekable();

    while let Some(header) = lines.next() {
        let mut header = header.split_whitespace();
        assert_eq!(header.next(), Some("case"), "expected a case header");
        let name = header.next().expect("case name").to_owned();
        let kind = header.next().expect("case kind").to_owned();
        let mut case = Case {
            name,
            kind,
            dimension: 0,
            points: Vec::new(),
            vertices: Vec::new(),
            simplices: Vec::new(),
            simplex_count: None,
        };

        loop {
            let line = lines.next().expect("a section or `end`");
            if line == "end" {
                break;
            }
            let mut section = line.split_whitespace();
            let key = section.next().expect("a section key");
            let first: usize = section.next().expect("a count").parse().unwrap();
            match key {
                "points" => {
                    case.dimension = first;
                    let count: usize = section.next().expect("a point count").parse().unwrap();
                    for _ in 0..count {
                        case.points.push(
                            lines
                                .next()
                                .expect("a point")
                                .split_whitespace()
                                .map(|v| v.parse().unwrap())
                                .collect(),
                        );
                    }
                }
                "vertices" => {
                    case.vertices = lines
                        .next()
                        .expect("the vertex list")
                        .split_whitespace()
                        .map(|v| v.parse().unwrap())
                        .collect();
                    assert_eq!(case.vertices.len(), first);
                }
                "triangles" | "facets" => {
                    for _ in 0..first {
                        let row: Vec<usize> = lines
                            .next()
                            .expect("a simplex")
                            .split_whitespace()
                            .map(|v| v.parse().unwrap())
                            .collect();
                        case.simplices.push([row[0], row[1], row[2]]);
                    }
                }
                "triangle_count" => case.simplex_count = Some(first),
                other => panic!("unknown section `{other}`"),
            }
        }
        cases.push(case);
    }
    cases
}

fn planar(case: &Case) -> Vec<[f64; 2]> {
    assert_eq!(case.dimension, 2);
    case.points.iter().map(|p| [p[0], p[1]]).collect()
}

fn spatial(case: &Case) -> Vec<[f64; 3]> {
    assert_eq!(case.dimension, 3);
    case.points.iter().map(|p| [p[0], p[1], p[2]]).collect()
}

fn sorted(simplices: &[[usize; 3]]) -> BTreeSet<[usize; 3]> {
    simplices
        .iter()
        .map(|s| {
            let mut sorted = *s;
            sorted.sort_unstable();
            sorted
        })
        .collect()
}

/// How many elements two sets do not share.
fn mismatch<T: Ord + Clone>(a: &BTreeSet<T>, b: &BTreeSet<T>) -> usize {
    a.symmetric_difference(b).count()
}

/// The distance to qhull on each of the eight cases: the extreme points, triangles or facets the two
/// codes do not share, or the difference in triangle count where the input has no unique answer.
#[test]
#[ignore = "comparator: the distance to qhull, not a verdict"]
fn construction_distance_to_qhull() {
    let cases = parse(FIXTURE);
    assert_eq!(cases.len(), 8, "the fixture lost a case");

    let mut distances = Vec::new();
    for case in &cases {
        let expected_vertices: BTreeSet<usize> = case.vertices.iter().copied().collect();
        let differ = match case.kind.as_str() {
            "hull2" => {
                let points = planar(case);
                let hull: BTreeSet<usize> = convex_hull_2d(&points).unwrap().into_iter().collect();
                mismatch(&hull, &expected_vertices)
            }
            "delaunay2" => {
                let points = planar(case);
                let mesh = Triangulation::new(&points).unwrap();
                mismatch(&sorted(mesh.triangles()), &sorted(&case.simplices))
            }
            "delaunay2_count" => {
                let points = planar(case);
                let mesh = Triangulation::new(&points).unwrap();
                let hull: BTreeSet<usize> = convex_hull_2d(&points).unwrap().into_iter().collect();
                mesh.triangles()
                    .len()
                    .abs_diff(case.simplex_count.expect("a triangle count"))
                    + mismatch(&hull, &expected_vertices)
            }
            "hull3" | "hull3_vertices" => {
                let points = spatial(case);
                let hull = ConvexHull::new(&points).unwrap();
                let vertices: BTreeSet<usize> = hull.vertices().iter().copied().collect();
                mismatch(&vertices, &expected_vertices)
                    + if case.kind == "hull3" {
                        mismatch(&sorted(hull.faces()), &sorted(&case.simplices))
                    } else {
                        0
                    }
            }
            other => panic!("unknown case kind `{other}`"),
        };
        println!(
            "{}: {differ} elements not shared with qhull on {} points",
            case.name,
            case.points.len()
        );
        #[expect(
            clippy::cast_precision_loss,
            reason = "a count of a few dozen elements"
        )]
        distances.push(differ as f64);
    }
    let max = distances.iter().copied().fold(0.0, f64::max);
    #[expect(clippy::cast_precision_loss, reason = "a count of eight cases")]
    let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
    println!(
        "comparison | qhull | not recorded | hull vertices, facets and Delaunay triangles not shared | element | {} | {max:.3e} | {rms:.3e}",
        distances.len()
    );
}
