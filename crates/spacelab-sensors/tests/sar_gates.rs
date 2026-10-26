//! `SideLookingSAR` and `look_side` against their defining vectors.
//!
//! `derivations/sar_incidence.mac` proves that on a sphere the cell an off-nadir look `eta`
//! reaches, at Earth-central angle `lam = asin(L sin eta) - eta`, is seen at incidence
//! `eta + lam`; `derivations/sar_incidence.gp` measures `acos(t/|d|)` there from the vectors at
//! 120 digits and prints the rows below, and evaluates one `look_side`.

#![allow(
    clippy::expect_used,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::approx_constant
)]

use spacelab_core::Vec3;
use spacelab_sensors::{AccessGeometry, Constraint, SideLookingSAR, look_side};

/// The sphere of `derivations/sar_incidence.gp`.
const R: f64 = 6_378_137.0;

/// `(h_m, eta_rad, theta_inc_rad)`, printed verbatim by `derivations/sar_incidence.gp`.
const INCIDENCE: [(f64, f64, f64); 15] = [
    (
        500000.0,
        2.617993877991494365385536e-1,
        2.828656868643873165944492e-1,
    ),
    (
        500000.0,
        3.490658503988659153847382e-1,
        3.777521755723463765408418e-1,
    ),
    (
        500000.0,
        5.061454830783555773078703e-1,
        5.501501181678142948871111e-1,
    ),
    (
        500000.0,
        6.457718232379019434617656e-1,
        7.062600535660059265861961e-1,
    ),
    (
        500000.0,
        8.028514559173916053848978e-1,
        8.878724105419859088067539e-1,
    ),
    (
        693000.0,
        2.617993877991494365385536e-1,
        2.910313523051895722327248e-1,
    ),
    (
        693000.0,
        3.490658503988659153847382e-1,
        3.889115345026728429368043e-1,
    ),
    (
        693000.0,
        5.061454830783555773078703e-1,
        5.674522627270602234878296e-1,
    ),
    (
        693000.0,
        6.457718232379019434617656e-1,
        7.304483674524658805495302e-1,
    ),
    (
        693000.0,
        8.028514559173916053848978e-1,
        9.231364167092115312055476e-1,
    ),
    (
        2000000.0,
        2.617993877991494365385536e-1,
        3.468926965668659931428968e-1,
    ),
    (
        2000000.0,
        3.490658503988659153847382e-1,
        4.659456085900555142602956e-1,
    ),
    (
        2000000.0,
        5.061454830783555773078703e-1,
        6.903822559670232935694877e-1,
    ),
    (
        2000000.0,
        6.457718232379019434617656e-1,
        9.116688269190288033567411e-1,
    ),
    (
        2000000.0,
        8.028514559173916053848978e-1,
        1.237301323836855192386697e0,
    ),
];

fn geometry(h: f64, lam: f64) -> AccessGeometry {
    AccessGeometry {
        satellite: Vec3::new((R + h) * lam.sin(), 0.0, (R + h) * lam.cos()),
        cell: Vec3::new(0.0, 0.0, R),
        zenith: Vec3::new(0.0, 0.0, 1.0),
    }
}

/// `1e-9` rad of central angle moves the satellite by at least 6.9 mm, far above `g`'s rounding.
const STEP: f64 = 1e-9;

/// (G2) The incidence the band is defined on is `eta + lam`: a band whose near edge, and one whose
/// far edge, is the 120-digit incidence has `g = 0` at that cell within `64 eps_mach (R + h)` m,
/// and `covered` flips across each edge.
#[test]
fn the_band_edges_are_the_incidence_of_the_look() {
    for (h, eta, inc) in INCIDENCE {
        let lam = inc - eta;
        let tol = 64.0 * f64::EPSILON * (R + h);
        let nearer = geometry(h, lam - STEP);
        let farther = geometry(h, lam + STEP);

        let near_edge = SideLookingSAR::new(inc, inc + 0.1).expect("band in range");
        let g = near_edge.g(&geometry(h, lam));
        assert!(g.abs() <= tol, "h {h}, eta {eta}: near-edge g = {g:.3e} m");
        assert!(!near_edge.covered(&nearer) && near_edge.covered(&farther));

        let far_edge = SideLookingSAR::new(inc - 0.1, inc).expect("band in range");
        let g = far_edge.g(&geometry(h, lam));
        assert!(g.abs() <= tol, "h {h}, eta {eta}: far-edge g = {g:.3e} m");
        assert!(far_edge.covered(&nearer) && !far_edge.covered(&farther));
    }
}

/// (G1) `look_side = (c - r) . (dhat x vhat) = -c . (r x v)/(|r| |v|)`, against the 120-digit
/// value for a non-circular state within `64 eps_mach |c - r|` m.
#[test]
fn look_side_is_minus_the_cell_along_the_orbit_normal() {
    let r = Vec3::new(6_800_000.0, 1_200_000.0, 300_000.0);
    let v = Vec3::new(-1_000.0, 6_500.0, 3_000.0);
    let c = Vec3::new(6_000_000.0, 2_000_000.0, -1_500_000.0);
    let want = 1.993602503676413031287660e6;
    let got = look_side(r, v, c);
    let tol = 64.0 * f64::EPSILON * (c - r).norm();
    assert!(
        (got - want).abs() <= tol,
        "look_side {got:.17e} m against {want:.17e} m"
    );
}
