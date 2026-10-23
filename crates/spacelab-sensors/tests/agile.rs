//! `AgileOptical`: the off-nadir slew cone intersected with visibility.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::many_single_char_names
)]

use spacelab_core::Vec3;
use spacelab_sensors::{
    AccessGeometry, AgileOptical, And, Cone, Constraint, ElevationMask, OrbitBound,
};

const RE: f64 = 6_378_137.0;
const MU: f64 = 3.986_004_418e14;

fn geom(sat: [f64; 3], cell: [f64; 3]) -> AccessGeometry {
    let c = Vec3::new(cell[0], cell[1], cell[2]);
    AccessGeometry {
        satellite: Vec3::new(sat[0], sat[1], sat[2]),
        cell: c,
        zenith: c.scale(1.0 / c.norm()), // spherical: outward normal is radial
    }
}

#[test]
fn agile_optical_is_the_cone_intersected_with_the_horizon() {
    let agile = AgileOptical::new(30.0_f64.to_radians()).unwrap();
    let composed = And(
        Cone::new(30.0_f64.to_radians()).unwrap(),
        ElevationMask::HORIZON,
    );

    // a spread of geometries: sub-point, off to the side, near the limb
    for cell in [
        [RE, 0.0, 0.0],
        [RE * 0.9, RE * 0.3, 0.1 * RE],
        [RE * 0.5, RE * 0.86, 0.0],
    ] {
        let g = geom([7_000e3, 0.0, 0.0], cell);
        assert!(
            (agile.g(&g) - composed.g(&g)).abs() < 1e-15,
            "g differs from the composition"
        );
        assert_eq!(
            agile.covered(&g),
            composed.covered(&g),
            "covered differs from the composition"
        );
    }
    let orbit = OrbitBound {
        semi_major_axis: 7_000e3,
        eccentricity: 0.0,
        mu: MU,
    };
    let (a, b) = (agile.max_rate(&orbit, 0.0), composed.max_rate(&orbit, 0.0));
    assert!((a - b).abs() < 1e-9, "max_rate differs: {a} vs {b}");
}

#[test]
fn the_horizon_term_rejects_the_far_side_a_bare_cone_would_accept() {
    // A cell antipodal to the sub-satellite point sits toward nadir — off-nadir ≈ 0 — so the bare
    // slew cone accepts it, but it is behind the Earth. AgileOptical must reject it.
    let sat = [7_000e3, 0.0, 0.0];
    let far_side = [-RE, 0.0, 0.0];
    let g = geom(sat, far_side);

    assert!(
        Cone::new(45.0_f64.to_radians()).unwrap().covered(&g),
        "the bare cone sees through the Earth"
    );
    assert!(
        !AgileOptical::new(45.0_f64.to_radians())
            .unwrap()
            .covered(&g),
        "AgileOptical must reject the far side"
    );
}

#[test]
fn a_cell_within_the_slew_limit_is_covered_and_one_beyond_it_is_not() {
    let sat = [7_000e3, 0.0, 0.0];
    let agile = AgileOptical::new(30.0_f64.to_radians()).unwrap();

    // sub-satellite point: off-nadir 0, well within the limit and above the horizon
    let below = geom(sat, [RE, 0.0, 0.0]);
    assert!(
        agile.covered(&below),
        "the sub-satellite point must be covered"
    );
    assert!(agile.g(&below) > 0.0);

    // a cell 60° of Earth-central angle away — past both the 30° slew and, here, the horizon
    let (c, s) = (60.0_f64.to_radians().cos(), 60.0_f64.to_radians().sin());
    let far = geom(sat, [RE * c, RE * s, 0.0]);
    assert!(
        !agile.covered(&far),
        "a cell 60° away exceeds a 30° slew limit"
    );
    assert!(agile.g(&far) < 0.0);
}
