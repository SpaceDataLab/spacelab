//! `geoloc` against the mathematics it codes: the ray-ellipsoid root, the geodetic latitude's
//! defining equation, the subpoint and the rotation sense.
//!
//! `derivations/ray_ellipsoid_geolocation.mac` proves that the coded `t` is the near root of the
//! quadratic `|t r − c|² = 1` in stretched axes, that the point at geodetic `(φ, h)` is a fixed point
//! of the coded latitude iteration, that the subpoint is on the ellipsoid under the geodetic
//! vertical, and that `rotate_about` is the right-handed Rodrigues rotation by `−θ`; it prints the
//! 40-digit values below, which `derivations/ray_ellipsoid_geolocation.gp` recomputes by
//! root-finding and by quaternions.

#![allow(
    clippy::expect_used,
    clippy::excessive_precision,
    clippy::unreadable_literal
)]

use spacelab_core::Vec3;
use spacelab_sensors::geoloc::{
    WGS84, geodetic_latitude, intersect_ellipsoid, rotate_about, subpoint,
};

fn v(c: [f64; 3]) -> Vec3 {
    Vec3::new(c[0], c[1], c[2])
}

fn assert_close(got: Vec3, want: [f64; 3], tolerance: f64, what: &str) {
    for (g, w) in [got.x, got.y, got.z].into_iter().zip(want) {
        assert!((g - w).abs() <= tolerance, "{what}: {got:?} vs {want:?}");
    }
}

/// `(origin, direction, hit)` in km, from `derivations/ray_ellipsoid_geolocation.mac`.
const HITS: [([f64; 3], [f64; 3], [f64; 3]); 3] = [
    (
        [5120.0, 3584.0, 2304.0],
        [-0.75, -0.5, -0.25],
        [
            4890.290815378335094847476252633982658003,
            3430.860543585556729898317501755988438668,
            2227.430271792778364949158750877994219334,
        ],
    ),
    (
        [7168.0, 0.0, 0.0],
        [-1.0, 0.25, 0.5],
        [
            6362.12128196360247518565889751358320676,
            201.46967950909938120358527562160419831,
            402.9393590181987624071705512432083966199,
        ],
    ),
    (
        [1024.0, -2048.0, 6656.0],
        [-0.125, 0.25, -1.0],
        [
            942.0112164896066729437312161930296918389,
            -1884.022432979213345887462432386059383678,
            6000.089731916853383549849729544237534712,
        ],
    ),
];

/// The hit is the near root of the ray's ellipsoid quadratic, and a ray whose discriminant is
/// negative has none. (G2)
///
/// The inputs are exact binary fractions, so the only error left is the rounding of the WGS-84 axes
/// and of the arithmetic: 1e-11 km is ten nanometres, about forty ulps of a 7000 km coordinate.
#[test]
fn the_ray_hit_is_the_near_root_of_the_ellipsoid_quadratic() {
    for (origin, direction, want) in HITS {
        let hit = intersect_ellipsoid(v(origin), v(direction), WGS84).expect("the ray hits");
        assert_close(hit, want, 1e-11, "hit");
    }
    // Tangent to the equator from 7168 km, along +y: the line stays outside the ellipsoid.
    assert!(intersect_ellipsoid(v([7168.0, 0.0, 0.0]), v([0.0, 1.0, 0.0]), WGS84).is_none());
}

/// `(phi_rad, point_km, subpoint_km)` for the points built at geodetic `(phi, lam, h)` by
/// `derivations/ray_ellipsoid_geolocation.mac`: `(11/16, 3/8, 830)`, `(−39/32, −5/2, 0)`,
/// `(3/2, 1, 35786)` and `(0, 1/4, 500)`.
const POINTS: [(f64, [f64; 3], [f64; 3]); 4] = [
    (
        0.6875,
        [
            5189.774376376714187859954821224595835099,
            2042.833117599752086115134390505359496098,
            4552.669169229801625033284288504557110374,
        ],
        [
            4592.897465840407941659330085441662559025,
            1807.88650305627562424890351061612326901,
            4025.945292817128108647527059715509814499,
        ],
    ),
    (
        -1.21875,
        [
            -1767.17869722729794810423723526096890889,
            -1320.121890033959005847309249555550562114,
            -5964.497471789410159029018412661337872095,
        ],
        [
            -1767.17869722729794810423723526096890889,
            -1320.121890033959005847309249555550562114,
            -5964.497471789410159029018412661337872095,
        ],
    ),
    (
        1.5,
        [
            1612.3070398063300776602343621570681776,
            2511.019438309857403672832230318964789423,
            42037.07724271363723638518156038863368714,
        ],
        [
            244.5849728831760277074734344590675461336,
            380.9185261027681394929370689141821087784,
            6340.721652100945370704669000719368356764,
        ],
    ),
    (
        0.0,
        [
            6664.312377527589183681955311197614020173,
            1701.678326095026579408480160226663255675,
            0.0,
        ],
        [
            6179.856166672266791609657586450519420271,
            1577.976346467765114610055807801968657728,
            0.0,
        ],
    ),
];

/// The iteration lands on the latitude the point was built at: the root of
/// `ρ sin φ − z cos φ = e² N sin φ cos φ`. (G2)
///
/// The point's coordinates are rounded to `f64` on the way in, which moves the latitude by about
/// an ulp of the coordinate over the radius, `1e-16`; the bound allows a few of those.
#[test]
fn the_geodetic_latitude_is_the_root_of_its_defining_equation() {
    for (phi, point, _) in POINTS {
        let got = geodetic_latitude(v(point), WGS84);
        assert!((got - phi).abs() < 1e-15, "phi {got} vs {phi}");
    }
}

/// The subpoint is the surface point at the same geodetic latitude and longitude — the point
/// minus `h` along the geodetic vertical — and it lies on the ellipsoid. (G2)
#[test]
fn the_subpoint_is_the_foot_of_the_geodetic_vertical() {
    for (_, point, want) in POINTS {
        let got = subpoint(v(point), WGS84);
        assert_close(got, want, 1e-11, "subpoint");
        let (a, b) = (WGS84.equatorial_radius, WGS84.polar_radius);
        let surface = (got.x / a).powi(2) + (got.y / a).powi(2) + (got.z / b).powi(2) - 1.0;
        assert!(surface.abs() < 1e-15, "off the ellipsoid by {surface}");
    }
}

/// `rotate_about` is the right-handed Rodrigues rotation by `−θ`, the sense the module documents.
/// (G1)
///
/// Two checks: about the cube diagonal by 120°, which permutes the axes exactly — `x̂ → ẑ` for
/// `−120°`, where the right-hand rule would give `ŷ` — and a generic axis and angle against the
/// 40-digit value.
#[test]
fn rotate_about_is_the_rodrigues_rotation_by_minus_the_angle() {
    let third_turn = 2.0 * core::f64::consts::PI / 3.0;
    let turned = rotate_about(v([1.0, 0.0, 0.0]), v([1.0, 1.0, 1.0]), third_turn);
    assert_close(turned, [0.0, 0.0, 1.0], 1e-15, "cube diagonal");
    let got = rotate_about(v([0.5, -1.0, 2.0]), v([1.0, 2.0, 2.0]), 0.75);
    assert_close(
        got,
        [
            -0.9229022158524859142860752937819913053789,
            -0.3554142093514988786851129648512779786391,
            2.066865317277741835828150611742273631329,
        ],
        1e-15,
        "generic rotation",
    );
}
