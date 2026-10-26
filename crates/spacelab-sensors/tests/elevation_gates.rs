//! `ElevationMask` against its defining vectors: the footprint edge in closed form.
//!
//! `derivations/elevation_footprint.mac` proves that on a sphere the cell at Earth-central angle
//! `lam = pi/2 - eps - asin(cos(eps)/L)` from the sub-satellite point sees the satellite at
//! exactly `eps`, at slant range `sqrt(r_s^2 - (R cos eps)^2) - R sin eps`;
//! `derivations/elevation_footprint.gp` evaluates both at 120 digits from the vectors and prints
//! the rows below.

#![allow(
    clippy::expect_used,
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::unreadable_literal
)]

use core::f64::consts::FRAC_PI_2;

use spacelab_core::Vec3;
use spacelab_sensors::{AccessGeometry, Constraint, ElevationMask, elevation::elevation};

/// The sphere of `derivations/elevation_footprint.gp`.
const R: f64 = 6_378_137.0;

/// `(h_m, eps_deg, lam_rad, slant_m)`, printed verbatim by `derivations/elevation_footprint.gp`
/// at 120 digits and cut to 25 here.
const EDGE: [(f64, f64, f64, f64); 30] = [
    (
        500000.0,
        0.0,
        3.836465158927483535429614e-1,
        2.574516847876509818821110e6,
    ),
    (
        500000.0,
        5.0,
        3.056998934906724181903116e-1,
        2.077956127651206684770193e6,
    ),
    (
        500000.0,
        10.0,
        2.451503638624737108305244e-1,
        1.695091195937326299460795e6,
    ),
    (
        500000.0,
        25.0,
        1.363978509447270706383602e-1,
        1.031941724402940872078776e6,
    ),
    (
        500000.0,
        60.0,
        4.148511543585046947449917e-2,
        5.705169373114309894241086e5,
    ),
    (
        500000.0,
        89.0,
        1.268866577929246180922568e-3,
        5.000706262019652767842537e5,
    ),
    (
        800000.0,
        0.0,
        4.766204303601609703857355e-1,
        3.293177675133851626299222e6,
    ),
    (
        800000.0,
        5.0,
        3.966721755450382565458527e-1,
        2.783874338172048624223427e6,
    ),
    (
        800000.0,
        10.0,
        3.307214218945261453213083e-1,
        2.366881948499214753418173e6,
    ),
    (
        800000.0,
        25.0,
        1.982823507535167623937772e-1,
        1.560165149823921885858012e6,
    ),
    (
        800000.0,
        60.0,
        6.323366609921286120570795e-2,
        9.071949850664925595563491e5,
    ),
    (
        800000.0,
        89.0,
        1.945326994092209336184957e-3,
        8.001082782689087494416961e5,
    ),
    (
        2000000.0,
        0.0,
        7.055061602738147758655616e-1,
        5.432545259820667154457158e6,
    ),
    (
        2000000.0,
        5.0,
        6.226957111323401686841098e-1,
        4.905021028122866106571436e6,
    ),
    (
        2000000.0,
        10.0,
        5.486278066801545006197275e-1,
        4.436744223438346062163337e6,
    ),
    (
        2000000.0,
        25.0,
        3.730342090130085731597148e-1,
        3.368999364706006845591003e6,
    ),
    (
        2000000.0,
        60.0,
        1.331086158594623247739329e-1,
        2.223823922975974990951504e6,
    ),
    (
        2000000.0,
        89.0,
        4.166673499323490627731265e-3,
        2.000231917992468702931602e6,
    ),
    (
        20200000.0,
        0.0,
        1.328454345768369118158457e0,
        2.580148706567123429140516e7,
    ),
    (
        20200000.0,
        5.0,
        1.242128445981080269493609e0,
        2.525158342292982178724592e7,
    ),
    (
        20200000.0,
        10.0,
        1.157675216727042519546714e0,
        2.471769558467162008371336e7,
    ),
    (
        20200000.0,
        25.0,
        9.152189019619758024667197e-1,
        2.324638999807212730376869e7,
    ),
    (
        20200000.0,
        60.0,
        4.033205637425577893488463e-1,
        2.086248905220265691926248e7,
    ),
    (
        20200000.0,
        89.0,
        1.326510714557205117900997e-2,
        2.020073831932083746704238e7,
    ),
    (
        35786000.0,
        0.0,
        1.418944163128946982106644e0,
        4.167893733486975552978051e7,
    ),
    (
        35786000.0,
        5.0,
        1.332260000810660042688024e0,
        4.112675299193481191772746e7,
    ),
    (
        35786000.0,
        10.0,
        1.246735699519634203079352e0,
        4.058609859089754635544579e7,
    ),
    (
        35786000.0,
        25.0,
        9.969343842183216890671829e-1,
        3.907049329323332888724998e7,
    ),
    (
        35786000.0,
        60.0,
        4.478918563279071367280304e-1,
        3.651973336241052094723698e7,
    ),
    (
        35786000.0,
        89.0,
        1.481327717845913950288381e-2,
        3.578682448595130711803641e7,
    ),
];

/// The polar cell, its zenith, and the satellite at central angle `lam` and altitude `h`.
fn geometry(h: f64, lam: f64) -> AccessGeometry {
    AccessGeometry {
        satellite: Vec3::new((R + h) * lam.sin(), 0.0, (R + h) * lam.cos()),
        cell: Vec3::new(0.0, 0.0, R),
        zenith: Vec3::new(0.0, 0.0, 1.0),
    }
}

/// (G2) On every edge row the defining vectors give back the mask: `elevation` returns `eps`
/// within `32 eps_mach (1 + 1/cos eps)` rad (the satellite's coordinates carry an ulp of
/// `R + h`, and `asin` amplifies by `1/cos eps`), the mask's `g = t - sin(eps)|d|` vanishes within
/// `64 eps_mach |d|` m, and `|d|` is the 120-digit slant range within `32 eps_mach (R + h)/|d|`
/// relative.
#[test]
fn the_mask_footprint_edge_is_where_the_closed_form_puts_it() {
    for (h, eps_deg, lam, slant) in EDGE {
        let eps = eps_deg.to_radians();
        let geom = geometry(h, lam);
        let d = geom.line_of_sight().norm();

        let got = elevation(geom.satellite, geom.cell, geom.zenith);
        let tol = 32.0 * f64::EPSILON * (1.0 + 1.0 / eps.cos());
        assert!(
            (got - eps).abs() <= tol,
            "h {h} m, mask {eps_deg} deg: elevation {got:.17e} rad, want {eps:.17e} (tol {tol:.1e})"
        );

        let g = ElevationMask::new(eps).expect("mask in range").g(&geom);
        assert!(
            g.abs() <= 64.0 * f64::EPSILON * d,
            "h {h} m, mask {eps_deg} deg: g = {g:.3e} m on the edge"
        );

        let rel = (d - slant).abs() / slant;
        assert!(
            rel <= 32.0 * f64::EPSILON * (R + h) / d,
            "h {h} m, mask {eps_deg} deg: |d| = {d:.17e} m, slant range {slant:.17e} m"
        );
    }
}

/// (G3) The exact special cases: straight overhead the elevation is `pi/2` to the ulp and the
/// horizon mask covers; at the antipode it is `-pi/2`; at the horizon tangency `cos(lam) = 1/L`
/// (the `eps = 0` rows) it is 0 within `32 eps_mach` rad and `g` of the horizon mask within
/// `64 eps_mach |d|`.
#[test]
fn overhead_antipode_and_horizon_are_exact() {
    for h in [500e3, 35_786e3] {
        let overhead = geometry(h, 0.0);
        assert_eq!(
            elevation(overhead.satellite, overhead.cell, overhead.zenith),
            FRAC_PI_2
        );
        assert!(ElevationMask::HORIZON.covered(&overhead));

        let antipode = geometry(h, core::f64::consts::PI);
        let below = elevation(antipode.satellite, antipode.cell, antipode.zenith);
        assert!(
            (below + FRAC_PI_2).abs() <= 2.0 * f64::EPSILON,
            "antipode at {below}"
        );
        assert!(!ElevationMask::HORIZON.covered(&antipode));

        let lam = (R / (R + h)).acos();
        let tangent = geometry(h, lam);
        let d = tangent.line_of_sight().norm();
        let at = elevation(tangent.satellite, tangent.cell, tangent.zenith);
        assert!(at.abs() <= 32.0 * f64::EPSILON, "tangency at {at:.3e} rad");
        assert!(ElevationMask::HORIZON.g(&tangent).abs() <= 64.0 * f64::EPSILON * d);
    }
}
