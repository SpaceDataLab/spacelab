//! Point in zone (`src/zone.rs`, Ф31.C): the bridges' reference, and the cases the model names.
//!
//! The C ABI (`crates/spacelab-capi/tests/access.rs`) and the Python bridge
//! (`crates/spacelab-python/tests/bindings.py`) read `fixtures/bridge_zone_contains_reference.tsv`,
//! which carries every vertex and point as the bits of its doubles and the API's answer for each
//! point. Each bridge must give every answer in it, and refuse every case it refuses with its own
//! typed error. This file makes the reference the Rust API's: it rebuilds it from
//! `spacelab::zone` and fails on any byte that differs.
//!
//! The cases the model names (G3): zones round either pole, a polar zone off the pole, across the
//! antimeridian, a hemisphere and more than one, an edge over a pole, vertex and edge hits, and
//! every refusal. The crossings themselves are held against the defining integrals at 50 digits in
//! the unit tests of `src/zone.rs` (G2).
//!
//! ```text
//! SPACELAB_WRITE_BRIDGE_ZONE=1 cargo test -p spacelab --test bridge_zone_contains
//! ```

#![expect(
    clippy::unwrap_used,
    reason = "a test fails loudly on a fixture or a case that does not hold"
)]

use core::f64::consts::FRAC_PI_2;
use std::fmt::Write as _;
use std::path::Path;

use spacelab::zone::{ANTIPODAL_MARGIN_RAD, Zone, ZoneError};
use spacelab_core::{Angle, Length};
use spacelab_earth::{GeodesicLine, SurfacePoint, geodesic_inverse};

const REFERENCE: &str = "tests/fixtures/bridge_zone_contains_reference.tsv";
const WRITE: &str = "SPACELAB_WRITE_BRIDGE_ZONE";

fn deg(lat: f64, lon: f64) -> SurfacePoint {
    SurfacePoint {
        latitude: Angle::from_radians(lat.to_radians()),
        longitude: Angle::from_radians(lon.to_radians()),
    }
}

fn points(vertices: &[(f64, f64)]) -> Vec<SurfacePoint> {
    vertices.iter().map(|&(lat, lon)| deg(lat, lon)).collect()
}

fn zone(vertices: &[(f64, f64)]) -> Zone {
    Zone::new(&points(vertices)).unwrap()
}

fn reversed(vertices: &[(f64, f64)]) -> Zone {
    let mut v = vertices.to_vec();
    v.reverse();
    zone(&v)
}

fn inside(z: &Zone, lat: f64, lon: f64) -> bool {
    z.contains(deg(lat, lon)).unwrap()
}

/// The zone holds `yes` and not `no`; its reversal, the complement, the other way round.
fn split(vertices: &[(f64, f64)], yes: &[(f64, f64)], no: &[(f64, f64)]) {
    let (z, r) = (zone(vertices), reversed(vertices));
    for &(lat, lon) in yes {
        assert!(
            inside(&z, lat, lon),
            "{vertices:?} should hold ({lat}, {lon})"
        );
        assert!(
            !inside(&r, lat, lon),
            "the reversal of {vertices:?} holds ({lat}, {lon})"
        );
    }
    for &(lat, lon) in no {
        assert!(!inside(&z, lat, lon), "{vertices:?} holds ({lat}, {lon})");
        assert!(
            inside(&r, lat, lon),
            "the reversal of {vertices:?} should hold ({lat}, {lon})"
        );
    }
}

/// Counter-clockwise seen from above the north pole is eastward. The edges bulge poleward: the one
/// from 0° to 90° E reaches 82.9° at 45° E, so (81°, 45°) is outside.
#[test]
fn a_zone_round_the_north_pole_holds_the_pole() {
    split(
        &[(80.0, 0.0), (80.0, 90.0), (80.0, 180.0), (80.0, -90.0)],
        &[
            (90.0, 0.0),
            (90.0, 123.0),
            (89.9, 45.0),
            (84.0, 45.0),
            (81.0, 0.0),
            (85.0, -135.0),
        ],
        &[(81.0, 45.0), (75.0, 10.0), (0.0, 0.0), (-90.0, 0.0)],
    );
}

/// Counter-clockwise seen from below the south pole is westward.
#[test]
fn a_zone_round_the_south_pole_holds_the_pole() {
    split(
        &[(-80.0, 0.0), (-80.0, -90.0), (-80.0, 180.0), (-80.0, 90.0)],
        &[(-90.0, 0.0), (-90.0, -77.0), (-89.0, 123.0), (-84.0, 45.0)],
        &[(-81.0, 45.0), (-70.0, 0.0), (0.0, 0.0), (90.0, 0.0)],
    );
}

/// A polar cap that does not reach the pole: a band of latitude between 70° and 75° over 60° of
/// longitude.
#[test]
fn a_polar_zone_off_the_pole() {
    split(
        &[(70.0, 10.0), (70.0, 70.0), (75.0, 70.0), (75.0, 10.0)],
        &[(74.5, 40.0), (72.0, 15.0), (74.0, 65.0)],
        &[
            (90.0, 0.0),
            (80.0, 40.0),
            (71.0, 40.0),
            (72.5, 5.0),
            (72.5, 75.0),
            (72.5, -140.0),
        ],
    );
}

#[test]
fn a_zone_across_the_antimeridian() {
    split(
        &[(-5.0, 175.0), (-5.0, -175.0), (5.0, -175.0), (5.0, 175.0)],
        &[
            (0.0, 180.0),
            (0.0, -180.0),
            (0.0, 179.9),
            (0.0, -179.9),
            (4.9, 175.1),
            (-4.9, -175.1),
        ],
        &[
            (0.0, 0.0),
            (0.0, 174.9),
            (0.0, -174.9),
            (5.1, 180.0),
            (-5.1, 180.0),
        ],
    );
}

/// Three equatorial edges of 120° bound the northern hemisphere exactly: equatorial geodesics are
/// shortest below `(1 − f)π`. A point 0.11 m off the equator is off the boundary.
#[test]
fn the_northern_hemisphere_is_a_zone() {
    let v = [(0.0, 0.0), (0.0, 120.0), (0.0, -120.0)];
    split(
        &v,
        &[(1e-6, 37.0), (45.0, -100.0), (90.0, 0.0)],
        &[(-1e-6, 37.0), (-45.0, -100.0), (-90.0, 0.0)],
    );
    let (z, r) = (zone(&v), reversed(&v));
    for lon in [0.0, 37.0, 120.0, 180.0, -60.0] {
        assert!(
            inside(&z, 0.0, lon) && inside(&r, 0.0, lon),
            "equator at {lon}"
        );
    }
}

/// The same edges pushed to 10° S dip to about 19.4° S between their ends: the zone holds the
/// north pole and more than a hemisphere, its complement a cap round the south pole.
#[test]
fn a_zone_larger_than_a_hemisphere() {
    split(
        &[(-10.0, 0.0), (-10.0, 120.0), (-10.0, -120.0)],
        &[
            (0.0, 60.0),
            (-15.0, 60.0),
            (-10.0, -60.0),
            (90.0, 0.0),
            (45.0, 170.0),
        ],
        &[(-25.0, 60.0), (-11.0, 0.0), (-90.0, 0.0), (-45.0, 170.0)],
    );
}

/// An edge over the north pole: (80°, 0°) to (80°, 180°) is the meridian through the pole. The
/// zone is on its left, the side of 90° W; a point on 90° E is 556 km from it, outside, and one on
/// the edge's own meridians is on the boundary, in the zone and its complement.
#[test]
fn an_edge_over_the_pole_bounds_the_zone_on_its_left() {
    let v = [(80.0, 0.0), (80.0, 180.0), (60.0, -90.0)];
    split(
        &v,
        &[(85.0, -90.0), (89.0, -30.0), (75.0, -90.0)],
        &[
            (85.0, 90.0),
            (89.0, 30.0),
            (70.0, 0.0),
            (70.0, 180.0),
            (55.0, -90.0),
        ],
    );
    let (z, r) = (zone(&v), reversed(&v));
    for (lat, lon) in [
        (85.0, 0.0),
        (85.0, 180.0),
        (90.0, 0.0),
        (90.0, 90.0),
        (80.0, 0.0),
    ] {
        assert!(
            inside(&z, lat, lon) && inside(&r, lat, lon),
            "({lat}, {lon})"
        );
    }
}

/// The boundary is shared: every vertex and every edge's midpoint is in the zone and in its
/// complement. A millimetre to the left of a midpoint is inside, a millimetre to the right is
/// outside, and the complement agrees the other way round.
#[test]
fn vertices_and_edges_belong_to_the_zone_and_to_its_complement() {
    let zones: [&[(f64, f64)]; 6] = [
        &[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
        &[(-5.0, 175.0), (-5.0, -175.0), (5.0, -175.0), (5.0, 175.0)],
        &[(80.0, 0.0), (80.0, 90.0), (80.0, 180.0), (80.0, -90.0)],
        &[(-10.0, 0.0), (-10.0, 120.0), (-10.0, -120.0)],
        &[(80.0, 0.0), (80.0, 180.0), (60.0, -90.0)],
        &[(-30.0, -60.0), (20.0, 30.0), (60.0, -20.0), (40.0, -100.0)],
    ];
    let mut checked = 0;
    for v in zones {
        let (zone_v, reverse) = (zone(v), reversed(v));
        for (i, &(lat, lon)) in v.iter().enumerate() {
            assert!(
                inside(&zone_v, lat, lon) && inside(&reverse, lat, lon),
                "{v:?} vertex {i}"
            );
            let (from, to) = (deg(lat, lon), {
                let (lat2, lon2) = v[(i + 1) % v.len()];
                deg(lat2, lon2)
            });
            let inverse = geodesic_inverse(from, to).unwrap();
            let mid = GeodesicLine::new(from, inverse.azimuth1)
                .unwrap()
                .position(Length::from_metres(0.5 * inverse.distance.metres()))
                .unwrap();
            assert!(
                zone_v.contains(mid.point).unwrap() && reverse.contains(mid.point).unwrap(),
                "{v:?} edge {i}"
            );
            for (side, held) in [(-FRAC_PI_2, true), (FRAC_PI_2, false)] {
                let off =
                    GeodesicLine::new(mid.point, Angle::from_radians(mid.azimuth.radians() + side))
                        .unwrap()
                        .position(Length::from_metres(1e-3))
                        .unwrap()
                        .point;
                assert_eq!(
                    zone_v.contains(off).unwrap(),
                    held,
                    "{v:?} edge {i} side {side}"
                );
                assert_eq!(
                    reverse.contains(off).unwrap(),
                    !held,
                    "{v:?} edge {i} side {side}"
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 22);
}

/// `ON_EDGE_M` is a micrometre: half of it off an edge is on it, twice is not.
#[test]
fn the_boundary_is_on_edge_m_wide() {
    let v = [(-30.0, -60.0), (20.0, 30.0), (60.0, -20.0), (40.0, -100.0)];
    let (z, r) = (zone(&v), reversed(&v));
    let g = geodesic_inverse(deg(-30.0, -60.0), deg(20.0, 30.0)).unwrap();
    let mid = GeodesicLine::new(deg(-30.0, -60.0), g.azimuth1)
        .unwrap()
        .position(Length::from_metres(0.37 * g.distance.metres()))
        .unwrap();
    let right = |d: f64| {
        GeodesicLine::new(
            mid.point,
            Angle::from_radians(mid.azimuth.radians() + FRAC_PI_2),
        )
        .unwrap()
        .position(Length::from_metres(d))
        .unwrap()
        .point
    };
    assert!(z.contains(right(0.5e-6)).unwrap());
    assert!(!z.contains(right(2e-6)).unwrap());
    assert!(r.contains(right(2e-6)).unwrap());
}

/// An edge on a meridian holds no point of the opposite half-meridian, half a turn away: the plane
/// of the meridian holds both, and a point there at a latitude the edge spans is as close to that
/// plane as a point on the edge. Points either side of the pole on a meridian edge's far side too.
#[test]
fn a_meridian_edge_holds_nothing_on_the_opposite_half_meridian() {
    let square = [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)];
    for (lat, lon) in [(0.5, 1.0), (-0.5, -1.0)] {
        assert!(inside(&zone(&square), lat, lon) && inside(&reversed(&square), lat, lon));
    }
    split(
        &square,
        &[(0.0, 0.0), (0.5, 0.5), (-0.5, -0.5)],
        &[
            (1.0, 179.0),
            (-1.0, -179.0),
            (0.5, 179.0),
            (-0.5, -179.0),
            (0.0, 181.0),
        ],
    );
    let to_the_pole = [(80.0, 10.0), (80.0, 20.0), (90.0, 10.0)];
    split(
        &to_the_pole,
        &[(85.0, 12.0), (89.999, 15.0)],
        &[(85.0, -170.0), (89.999, -170.0)],
    );
}

#[test]
fn every_refusal_is_typed() {
    let refused = |v: &[(f64, f64)]| Zone::new(&points(v)).unwrap_err();
    assert_eq!(Zone::new(&[]).unwrap_err(), ZoneError::TooFewVertices);
    assert_eq!(
        refused(&[(0.0, 0.0), (1.0, 1.0)]),
        ZoneError::TooFewVertices
    );
    assert_eq!(
        refused(&[(0.0, 0.0), (f64::NAN, 1.0), (1.0, 0.0)]),
        ZoneError::Vertex { index: 1 }
    );
    assert_eq!(
        refused(&[(0.0, 0.0), (0.0, 1.0), (91.0, 0.0)]),
        ZoneError::Vertex { index: 2 }
    );
    assert_eq!(
        refused(&[(0.0, f64::INFINITY), (0.0, 1.0), (1.0, 0.0)]),
        ZoneError::Vertex { index: 0 }
    );
    // Coincident consecutive vertices: an edge of length zero has no direction.
    assert_eq!(
        refused(&[(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]),
        ZoneError::Edge { index: 0 }
    );
    // The margin is 2fπ, 1.2° on the equator: 179° apart is refused, 178° is a zone.
    assert!((ANTIPODAL_MARGIN_RAD.to_degrees() - 1.2070).abs() < 1e-4);
    assert_eq!(
        refused(&[(0.0, 0.0), (0.0, 179.0), (10.0, 90.0)]),
        ZoneError::Edge { index: 0 }
    );
    assert_eq!(
        refused(&[(10.0, 90.0), (0.0, 0.0), (0.0, 179.0)]),
        ZoneError::Edge { index: 1 }
    );
    assert!(Zone::new(&points(&[(0.0, 0.0), (0.0, 178.0), (10.0, 90.0)])).is_ok());
    // A bow tie: edges 0 and 2 cross.
    assert_eq!(
        refused(&[(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]),
        ZoneError::SelfIntersecting {
            first: 0,
            second: 2
        }
    );
    // A vertex on a non-adjacent edge: (0°, 2°) touches the equatorial edge 0.
    assert_eq!(
        refused(&[(0.0, 0.0), (0.0, 4.0), (2.0, 4.0), (0.0, 2.0), (2.0, 0.0)]),
        ZoneError::SelfIntersecting {
            first: 0,
            second: 2
        }
    );
    // An edge doubling back over its predecessor along the meridian.
    assert_eq!(
        refused(&[(0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),
        ZoneError::SelfIntersecting {
            first: 0,
            second: 1
        }
    );
    let z = zone(&[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]);
    for (lat, lon) in [(f64::NAN, 0.0), (0.0, f64::NEG_INFINITY), (-90.5, 0.0)] {
        assert_eq!(z.contains(deg(lat, lon)).unwrap_err(), ZoneError::Point);
    }
}

fn bits(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

fn rad(lat: f64, lon: f64) -> SurfacePoint {
    SurfacePoint {
        latitude: Angle::from_radians(lat),
        longitude: Angle::from_radians(lon),
    }
}

/// One zone of the reference: its vertices and any points beyond the generated ones, in degrees.
struct Case {
    name: &'static str,
    vertices: Vec<(f64, f64)>,
    extra: Vec<(f64, f64)>,
}

fn case(name: &'static str, vertices: &[(f64, f64)]) -> Case {
    Case {
        name,
        vertices: vertices.to_vec(),
        extra: Vec::new(),
    }
}

/// The G3 zones, each also reversed, and one case per refusal kind.
fn cases() -> Vec<Case> {
    let zones: [(&str, &[(f64, f64)]); 9] = [
        (
            "square-1deg",
            &[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
        ),
        (
            "north-polar-cap",
            &[(80.0, 0.0), (80.0, 90.0), (80.0, 180.0), (80.0, -90.0)],
        ),
        (
            "south-polar-cap",
            &[(-80.0, 0.0), (-80.0, -90.0), (-80.0, 180.0), (-80.0, 90.0)],
        ),
        (
            "polar-band",
            &[(70.0, 10.0), (70.0, 70.0), (75.0, 70.0), (75.0, 10.0)],
        ),
        (
            "antimeridian",
            &[(-5.0, 175.0), (-5.0, -175.0), (5.0, -175.0), (5.0, 175.0)],
        ),
        (
            "northern-hemisphere",
            &[(0.0, 0.0), (0.0, 120.0), (0.0, -120.0)],
        ),
        (
            "hemisphere-plus",
            &[(-10.0, 0.0), (-10.0, 120.0), (-10.0, -120.0)],
        ),
        (
            "edge-over-the-pole",
            &[(80.0, 0.0), (80.0, 180.0), (60.0, -90.0)],
        ),
        (
            "quadrilateral",
            &[(-30.0, -60.0), (20.0, 30.0), (60.0, -20.0), (40.0, -100.0)],
        ),
    ];
    let reversed_names = [
        "square-1deg-reversed",
        "north-polar-cap-reversed",
        "south-polar-cap-reversed",
        "polar-band-reversed",
        "antimeridian-reversed",
        "northern-hemisphere-reversed",
        "hemisphere-plus-reversed",
        "edge-over-the-pole-reversed",
        "quadrilateral-reversed",
    ];
    let mut out = Vec::new();
    for ((name, v), reversed) in zones.into_iter().zip(reversed_names) {
        out.push(case(name, v));
        let mut r = v.to_vec();
        r.reverse();
        out.push(case(reversed, &r));
    }
    out.push(case("two-vertices", &[(0.0, 0.0), (1.0, 1.0)]));
    out.push(case(
        "vertex-not-finite",
        &[(0.0, 0.0), (f64::NAN, 1.0), (1.0, 0.0)],
    ));
    out.push(case(
        "vertex-past-the-pole",
        &[(0.0, 0.0), (0.0, 1.0), (91.0, 0.0)],
    ));
    out.push(case(
        "coincident-vertices",
        &[(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)],
    ));
    out.push(case(
        "nearly-antipodal-edge",
        &[(0.0, 0.0), (0.0, 179.0), (10.0, 90.0)],
    ));
    out.push(case(
        "bow-tie",
        &[(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)],
    ));
    out.push(Case {
        name: "point-not-finite",
        vertices: vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        extra: vec![(0.5, f64::INFINITY)],
    });
    out
}

/// The points asked about a case, in radians: a 7 x 12 grid, both poles, every vertex, and where
/// the zone is one, every edge's midpoint and the points a millimetre either side of it.
fn points_of(case: &Case) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for i in 0..7_i32 {
        for j in 0..12_i32 {
            let (lat, lon) = (f64::from(25 * i - 75), f64::from(30 * j - 150));
            out.push((lat.to_radians(), lon.to_radians()));
        }
    }
    out.push((FRAC_PI_2, 0.0));
    out.push((-FRAC_PI_2, 0.0));
    let vertices: Vec<SurfacePoint> = case
        .vertices
        .iter()
        .map(|&(lat, lon)| deg(lat, lon))
        .collect();
    out.extend(
        vertices
            .iter()
            .map(|v| (v.latitude.radians(), v.longitude.radians())),
    );
    if Zone::new(&vertices).is_ok() {
        for (i, &a) in vertices.iter().enumerate() {
            let b = vertices[(i + 1) % vertices.len()];
            let g = geodesic_inverse(a, b).unwrap();
            let mid = GeodesicLine::new(a, g.azimuth1)
                .unwrap()
                .position(Length::from_metres(0.5 * g.distance.metres()))
                .unwrap();
            out.push((mid.point.latitude.radians(), mid.point.longitude.radians()));
            for side in [-FRAC_PI_2, FRAC_PI_2] {
                let off =
                    GeodesicLine::new(mid.point, Angle::from_radians(mid.azimuth.radians() + side))
                        .unwrap()
                        .position(Length::from_metres(1e-3))
                        .unwrap()
                        .point;
                out.push((off.latitude.radians(), off.longitude.radians()));
            }
        }
    }
    out.extend(
        case.extra
            .iter()
            .map(|&(lat, lon)| (lat.to_radians(), lon.to_radians())),
    );
    out
}

/// The refusal kinds, one name per typed error.
fn kind(e: &ZoneError) -> &'static str {
    match e {
        ZoneError::TooFewVertices => "vertices",
        ZoneError::Vertex { .. } => "vertex",
        ZoneError::Edge { .. } => "edge",
        ZoneError::SelfIntersecting { .. } => "intersecting",
        ZoneError::Point => "point",
    }
}

/// A case's answers, point by point, as the bridges batch them: the zone is built first, then each
/// point is asked; the first refusal refuses the batch.
fn run(vertices: &[(f64, f64)], points: &[(f64, f64)]) -> Result<Vec<bool>, ZoneError> {
    let v: Vec<SurfacePoint> = vertices.iter().map(|&(lat, lon)| rad(lat, lon)).collect();
    let zone = Zone::new(&v)?;
    points
        .iter()
        .map(|&(lat, lon)| zone.contains(rad(lat, lon)))
        .collect()
}

/// The whole file, from the cases and the Rust API.
fn reference() -> String {
    let mut out = String::new();
    out.push_str(
        "# The Rust API's point-in-zone answers (spacelab::zone::Zone::contains), exactly \
         (Ф31.C).\n\
         # Written by crates/spacelab/tests/bridge_zone_contains.rs; the C ABI and the Python \
         bridge must reproduce it. A double is the 16 hex digits of its bits.\n\
         # Angles are geodetic radians on WGS 84, as every bridge takes them. A zone lies on the \
         left of each edge, the shortest geodesic to the next vertex; a refused case answers no \
         point. Rows, tab-separated:\n\
         #   zone    <case> <vertex count> <point count>\n\
         #   vertex  <case> <index> <latitude rad> <longitude rad>\n\
         #   point   <case> <index> <latitude rad> <longitude rad> <inside: 1 or 0, - if refused>\n\
         #   refused <case> <kind>\n",
    );
    for case in cases() {
        let vertices: Vec<(f64, f64)> = case
            .vertices
            .iter()
            .map(|&(lat, lon)| (lat.to_radians(), lon.to_radians()))
            .collect();
        let points = points_of(&case);
        writeln!(
            out,
            "zone\t{}\t{}\t{}",
            case.name,
            vertices.len(),
            points.len()
        )
        .unwrap();
        for (i, &(lat, lon)) in vertices.iter().enumerate() {
            writeln!(
                out,
                "vertex\t{}\t{i}\t{}\t{}",
                case.name,
                bits(lat),
                bits(lon)
            )
            .unwrap();
        }
        let answers = run(&vertices, &points);
        for (i, &(lat, lon)) in points.iter().enumerate() {
            let answer = answers
                .as_ref()
                .map_or("-", |a| if a[i] { "1" } else { "0" });
            writeln!(
                out,
                "point\t{}\t{i}\t{}\t{}\t{answer}",
                case.name,
                bits(lat),
                bits(lon)
            )
            .unwrap();
        }
        if let Err(e) = answers {
            writeln!(out, "refused\t{}\t{}", case.name, kind(&e)).unwrap();
        }
    }
    out
}

/// The committed file is the Rust API's answer, byte for byte.
#[test]
fn the_zone_reference_is_the_rust_api_on_its_cases() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(REFERENCE);
    let want = reference();
    if std::env::var_os(WRITE).is_some() {
        std::fs::write(&path, &want).unwrap();
    }
    let have = std::fs::read_to_string(&path).unwrap_or_default();
    if have != want {
        let line = have
            .lines()
            .zip(want.lines())
            .position(|(a, b)| a != b)
            .map_or(have.lines().count().min(want.lines().count()) + 1, |i| {
                i + 1
            });
        panic!(
            "{REFERENCE} is not the Rust API's answer from line {line}: rerun with {WRITE}=1 after \
             a deliberate change, and read the diff before committing it"
        );
    }
}

/// Every refusal kind is in the reference, so a bridge that mismaps one is caught, and the
/// answered cases hold both answers.
#[test]
fn the_zone_reference_refuses_every_kind_and_answers_both_ways() {
    let text = reference();
    let kinds: std::collections::BTreeSet<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("refused\t"))
        .map(|l| l.split('\t').nth(1).unwrap())
        .collect();
    assert_eq!(
        kinds.into_iter().collect::<Vec<_>>(),
        ["edge", "intersecting", "point", "vertex", "vertices"],
        "the reference does not refuse every kind"
    );
    let answers = |a: &str| {
        text.lines()
            .filter(|l| l.starts_with("point\t") && l.ends_with(a))
            .count()
    };
    assert!(answers("\t1") > 500 && answers("\t0") > 500);
}
