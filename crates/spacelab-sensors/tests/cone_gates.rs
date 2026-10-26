//! `Cone` and `AgileOptical` against their defining vectors: the footprint edges in closed form.
//!
//! `derivations/cone_footprint.mac` proves that on a sphere the near-side cone edge is
//! `lam_c = asin(L sin alpha) - alpha`, the horizon edge `cos(lam_h) = 1/L`, and that the two meet
//! at the tangency `L sin alpha = 1`; `derivations/cone_footprint.gp` measures the off-nadir angle
//! and the rise from the vectors at 120 digits and prints the rows below.

#![allow(
    clippy::expect_used,
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::approx_constant
)]

use spacelab_core::Vec3;
use spacelab_sensors::{
    AccessGeometry, AgileOptical, Boresight, Cone, Constraint, ElevationMask, OrbitBound,
};

/// The sphere of `derivations/cone_footprint.gp`.
const R: f64 = 6_378_137.0;

/// `(h_m, alpha_rad, lam_cone_rad, lam_horizon_rad)`, printed verbatim by
/// `derivations/cone_footprint.gp`; `lam_cone = -1` where the cone overhangs the limb, and the
/// last row of each altitude is the tangency `alpha = asin(1/L)`, where the two edges coincide.
const EDGE: [(f64, f64, f64, f64); 20] = [
    (
        500000.0,
        1.745329251994329576923691e-1,
        1.384009435577161240458444e-2,
        3.836465158927483535429614e-1,
    ),
    (
        500000.0,
        5.235987755982988730771072e-1,
        4.588385006583551660819317e-2,
        3.836465158927483535429614e-1,
    ),
    (
        500000.0,
        8.726646259971647884618454e-1,
        9.948113263024523395767243e-2,
        3.836465158927483535429614e-1,
    ),
    (
        500000.0,
        1.047197551196597746154214e0,
        1.580159339132119889995190e-1,
        3.836465158927483535429614e-1,
    ),
    (
        500000.0,
        1.187149810902148265688360e0,
        3.836465158927483535429614e-1,
        3.836465158927483535429614e-1,
    ),
    (
        800000.0,
        1.745329251994329576923691e-1,
        2.216153665904195045791356e-2,
        4.766204303601609703857355e-1,
    ),
    (
        800000.0,
        5.235987755982988730771072e-1,
        7.406678147827017078767903e-2,
        4.766204303601609703857355e-1,
    ),
    (
        800000.0,
        8.726646259971647884618454e-1,
        1.667904146445843594633793e-1,
        4.766204303601609703857355e-1,
    ),
    (
        800000.0,
        1.047197551196597746154214e0,
        2.979522074973084294313796e-1,
        4.766204303601609703857355e-1,
    ),
    (
        800000.0,
        1.094175896434735648845586e0,
        4.766204303601609703857355e-1,
        4.766204303601609703857355e-1,
    ),
    (
        2000000.0,
        1.745329251994329576923691e-1,
        5.559208823378536771426974e-2,
        7.055061602738147758655616e-1,
    ),
    (
        2000000.0,
        5.235987755982988730771072e-1,
        1.929493294972777569421778e-1,
        7.055061602738147758655616e-1,
    ),
    (
        2000000.0,
        8.726646259971647884618454e-1,
        -1.0,
        7.055061602738147758655616e-1,
    ),
    (
        2000000.0,
        1.047197551196597746154214e0,
        -1.0,
        7.055061602738147758655616e-1,
    ),
    (
        2000000.0,
        8.652901665210818433657601e-1,
        7.055061602738147758655616e-1,
        7.055061602738147758655616e-1,
    ),
    (
        35786000.0,
        1.745329251994329576923691e-1,
        -1.0,
        1.418944163128946982106644e0,
    ),
    (
        35786000.0,
        5.235987755982988730771072e-1,
        -1.0,
        1.418944163128946982106644e0,
    ),
    (
        35786000.0,
        8.726646259971647884618454e-1,
        -1.0,
        1.418944163128946982106644e0,
    ),
    (
        35786000.0,
        1.047197551196597746154214e0,
        -1.0,
        1.418944163128946982106644e0,
    ),
    (
        35786000.0,
        1.518521636659496371246772e-1,
        1.418944163128946982106644e0,
        1.418944163128946982106644e0,
    ),
];

/// `1/(1 - e^2) = 1/(1 - f)^2` for `f = 1/298.257223563`, exact in `derivations/cone_footprint.gp`.
const INVERSE_ONE_MINUS_E2: f64 = 1.006739496742276434954782;

fn geometry(h: f64, lam: f64) -> AccessGeometry {
    AccessGeometry {
        satellite: Vec3::new((R + h) * lam.sin(), 0.0, (R + h) * lam.cos()),
        cell: Vec3::new(0.0, 0.0, R),
        zenith: Vec3::new(0.0, 0.0, 1.0),
    }
}

/// How far either side of an edge the inside/outside verdict is asserted: `1e-9` rad of central
/// angle moves the satellite by at least 6.9 mm, six orders above the rounding of `g`.
const STEP: f64 = 1e-9;

/// (G2) On every cone edge the off-nadir angle is the half-angle: `g` of the cone about the
/// geocentric nadir, and about the same axis supplied as `Boresight::Fixed`, vanishes within
/// `64 eps_mach (R + h)` m (the division-free `u~ = n2 - c.r` cancels terms of order
/// `(R + h)^2`), and `covered` flips across it.
#[test]
fn the_cone_footprint_edge_is_where_the_closed_form_puts_it() {
    for (h, alpha, lam_c, lam_h) in EDGE {
        if lam_c < 0.0 {
            continue;
        }
        let tol = 64.0 * f64::EPSILON * (R + h);
        let geom = geometry(h, lam_c);
        let axis = geom.satellite.scale(-1.0);
        for cone in [
            Cone::new(alpha).expect("half-angle in range"),
            Cone::pointed(alpha, Boresight::Fixed(axis)).expect("half-angle in range"),
        ] {
            let g = cone.g(&geom);
            assert!(
                g.abs() <= tol,
                "h {h} m, alpha {alpha} rad, {:?}: g = {g:.3e} m on the edge",
                cone.boresight()
            );
            // At the tangency the off-nadir angle peaks at the limb, a double root: the bare cone
            // covers both sides of it (it sees through the Earth), so only `g` is asserted there.
            if lam_c == lam_h {
                continue;
            }
            let inside = geometry(h, lam_c - STEP);
            let outside = geometry(h, lam_c + STEP);
            let fixed = |g: &AccessGeometry| match cone.boresight() {
                Boresight::GeocentricNadir => cone,
                Boresight::Fixed(_) => {
                    Cone::pointed(alpha, Boresight::Fixed(g.satellite.scale(-1.0)))
                        .expect("half-angle in range")
                }
            };
            assert!(
                fixed(&inside).covered(&inside),
                "h {h}, alpha {alpha}: inside"
            );
            assert!(
                !fixed(&outside).covered(&outside),
                "h {h}, alpha {alpha}: outside"
            );
        }
    }
}

/// (G3) At the tangency `L sin(alpha) = 1` the cone edge is `pi/2 - alpha` and coincides with
/// the horizon edge `acos(1/L)`: both the cone's and the horizon mask's `g` vanish there within
/// `64 eps_mach (R + h)` m.
#[test]
fn at_the_tangency_the_cone_edge_is_the_horizon() {
    let mut seen = 0;
    for (h, alpha, lam_c, lam_h) in EDGE {
        if lam_c != lam_h {
            continue;
        }
        seen += 1;
        assert!((lam_c - (core::f64::consts::FRAC_PI_2 - alpha)).abs() <= 4.0 * f64::EPSILON);
        let geom = geometry(h, lam_h);
        let tol = 64.0 * f64::EPSILON * (R + h);
        let cone = Cone::new(alpha).expect("half-angle in range").g(&geom);
        let horizon = ElevationMask::HORIZON.g(&geom);
        assert!(
            cone.abs() <= tol && horizon.abs() <= tol,
            "h {h}: {cone:.3e}, {horizon:.3e}"
        );
    }
    assert_eq!(seen, 4, "one tangency row per altitude");
}

/// (G2) The agile footprint `Cone(alpha) and HORIZON` ends at the nearer of the two edges --
/// `lam_c` while the cone is in front of the limb, `lam_h` once it overhangs it: `g` vanishes
/// there within `64 eps_mach (R + h)` m and `covered` flips across it. At the tangency the cone
/// term touches zero quadratically, so the flip is asserted `1e-4` rad either side instead.
#[test]
fn the_agile_footprint_edge_is_the_nearer_of_the_cone_and_the_horizon() {
    for (h, alpha, lam_c, lam_h) in EDGE {
        let agile = AgileOptical::new(alpha).expect("slew limit in range");
        let edge = if lam_c < 0.0 { lam_h } else { lam_c.min(lam_h) };
        let step = if lam_c == lam_h { 1e-4 } else { STEP };
        let g = agile.g(&geometry(h, edge));
        assert!(
            g.abs() <= 64.0 * f64::EPSILON * (R + h),
            "h {h} m, alpha {alpha} rad: g = {g:.3e} m on the edge"
        );
        assert!(
            agile.covered(&geometry(h, edge - step)),
            "h {h}, alpha {alpha}: inside"
        );
        assert!(
            !agile.covered(&geometry(h, edge + step)),
            "h {h}, alpha {alpha}: outside"
        );
    }
}

/// (G3) A `Boresight::Fixed` cone's rate bound scales the nadir-turn term by exactly
/// `1/(1 - e^2) = 1/(1 - f)^2` on WGS-84: the extra over the geocentric bound, divided by that
/// term, is the 120-digit constant within 8 ulp.
#[test]
fn the_fixed_boresight_turn_term_is_one_over_one_minus_e_squared() {
    let orbit = OrbitBound {
        semi_major_axis: 7_000e3,
        eccentricity: 0.001,
        mu: 3.986_004_418e14,
    };
    let alpha = 30.0_f64.to_radians();
    let geocentric = Cone::new(alpha).expect("half-angle in range");
    let fixed = Cone::pointed(alpha, Boresight::Fixed(Vec3::new(0.0, 0.0, -1.0)))
        .expect("half-angle in range");
    let v = orbit.max_relative_speed(0.0);
    let turn = 2.0 * orbit.apogee_radius() * v / orbit.perigee_radius();
    let base = (1.0 + geocentric.cos_half()) * v;
    let ratio = (fixed.max_rate(&orbit, 0.0) - base) / (geocentric.max_rate(&orbit, 0.0) - base);
    assert!(
        (ratio - INVERSE_ONE_MINUS_E2).abs() <= 8.0 * f64::EPSILON,
        "ratio {ratio:.17e} against {INVERSE_ONE_MINUS_E2:.17e} (turn term {turn:.3e} m/s)"
    );
}
