//! WGS-84 geodesy: the derived constants and both directions of the geodetic map against the
//! mathematics that defines them, then the forward/inverse round-trip, the geocentric singularity,
//! and the batch entry.
//!
//! The gates are `the_derived_constants_are_the_exact_functions_of_a_and_f` (G1) and
//! `the_inverse_solves_the_defining_equation_of_the_geodetic_latitude` (G2), with
//! `the_forward_map_is_the_ellipsoid_parametrisation_at_80_digits` (G1) beside them. Their values
//! come from `derivations/wgs84_geodesy.gp`, which solves the defining equation rather than
//! running Vermeille's algorithm; `derivations/wgs84_geodesy.mac` shows the forward map satisfies
//! that equation and Vermeille's quartic identically.

// The reference rows are pasted verbatim as the derivation prints them, without separators; some
// of its angles are π/2 and π/4 to the last digit because the point is on the axis or the diagonal.
#![allow(
    clippy::unwrap_used,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::approx_constant
)]

use spacelab_core::{Angle, Length, Vec3};
use spacelab_earth::{
    EQUATORIAL_RADIUS, FLATTENING, Geodetic, GeodeticError, POLAR_RADIUS, geodetic_nadir,
    geodetic_to_itrf, itrf_to_geodetic, itrf_to_geodetic_into,
};
use spacelab_frames::{Itrf, Position};

fn geodetic(lat_deg: f64, lon_deg: f64, height_m: f64) -> Geodetic {
    Geodetic {
        latitude: Angle::from_degrees(lat_deg),
        longitude: Angle::from_degrees(lon_deg),
        height: Length::from_metres(height_m),
    }
}

#[test]
fn geodetic_and_itrf_round_trip_at_realistic_points() {
    // Vermeille's inverse is a closed form, so the round-trip must return to the ellipsoid point it
    // started from — at cities, near the pole, and near the equator/antimeridian.
    for (lat, lon, h) in [
        (48.8566, 2.3522, 35.0),    // Paris
        (-33.8688, 151.2093, 58.0), // Sydney
        (89.9, 12.0, 100.0),        // near the North Pole
        (0.0, 179.0, 0.0),          // equator, near the antimeridian
        (0.0, 0.0, 20_200_000.0),   // a MEO-altitude sub-point
    ] {
        let back = itrf_to_geodetic(geodetic_to_itrf(geodetic(lat, lon, h))).unwrap();
        assert!(
            (back.latitude.degrees() - lat).abs() < 1e-9,
            "lat at ({lat}, {lon})"
        );
        assert!(
            (back.longitude.degrees() - lon).abs() < 1e-9,
            "lon at ({lat}, {lon})"
        );
        assert!(
            (back.height.metres() - h).abs() < 1e-6,
            "height at ({lat}, {lon})"
        );
    }
}

#[test]
fn a_point_in_the_geocentric_singularity_has_no_geodetic_coordinates() {
    let centre = Position::<Itrf>::new(Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(itrf_to_geodetic(centre), Err(GeodeticError::NearGeocentre));
    assert_ne!(GeodeticError::NearGeocentre.to_string(), "");
}

#[test]
fn the_batch_converter_maps_each_slot_independently() {
    let paris = geodetic_to_itrf(geodetic(48.8566, 2.3522, 35.0));
    let centre = Position::<Itrf>::new(Vec3::new(0.0, 0.0, 0.0));
    let positions = [paris, centre];
    let mut out = [Err(GeodeticError::NearGeocentre); 2];
    itrf_to_geodetic_into(&positions, &mut out);
    assert!(out[0].is_ok());
    assert_eq!(out[1], Err(GeodeticError::NearGeocentre)); // a decayed slot does not spoil the batch
}

#[test]
#[should_panic(expected = "output slice holds")]
fn the_batch_converter_refuses_a_short_output() {
    let p = geodetic_to_itrf(geodetic(0.0, 0.0, 0.0));
    let positions = [p, p];
    let mut out: [Result<Geodetic, GeodeticError>; 1] = [Err(GeodeticError::NearGeocentre)];
    itrf_to_geodetic_into(&positions, &mut out);
}

/// `b = a(1 − f)` and `f = 1/298.257 223 563` at 40 digits, with `a` and `1/f` taken as the exact
/// rationals the definition decrees (`derivations/wgs84_geodesy.gp`, `constants:` lines).
const EXACT_B: f64 = 6_356_752.314_245_179_497_563_966_599_633_655_156_798;
const EXACT_F: f64 = 0.003_352_810_664_747_480_719_845_528_618_520_555_955_733;

/// (G1) The published constants are the exact functions of the two defining ones, `a` and `1/f`.
///
/// `f` is `1/298.257223563` computed in `f64`: the decimal rounds once to a double and the quotient
/// rounds again, so one ulp is the most it can be off. `b = a(1 − f)` adds two roundings on top, so
/// two ulps. `e² = f(2 − f)` is private; it reaches the outputs below, where a wrong `e²` misses the
/// latitude by the deflection it sets, 1e-3 rad, against a bound of 1e-15.
#[test]
fn the_derived_constants_are_the_exact_functions_of_a_and_f() {
    assert!(
        (EQUATORIAL_RADIUS - 6_378_137.0).abs() == 0.0,
        "a is exact by decree"
    );
    let f_ulps = (FLATTENING - EXACT_F).abs() / (EXACT_F * f64::EPSILON);
    let b_ulps = (POLAR_RADIUS - EXACT_B).abs() / (EXACT_B * f64::EPSILON);
    println!("WGS-84 constants: f off by {f_ulps:.2} ulp, b off by {b_ulps:.2} ulp");
    assert!(f_ulps <= 1.0, "f is {f_ulps:.2} ulp from 1/298.257223563");
    assert!(b_ulps <= 2.0, "b is {b_ulps:.2} ulp from a(1 - f)");
}

/// `(x, y, z, φ, λ, h)`: integer-metre ECEF points and the geodetic coordinates that solve
/// `ρ sin φ − z cos φ = e² N(φ) sin φ cos φ` for them at 80 digits, `h = ρ cos φ + z sin φ − a W(φ)`.
/// Equator, both poles and the near-axis points beside them, mid latitudes, LEO to GEO and beyond,
/// below the surface, and 112 km from the centre, outside the 43 km lens the inverse refuses.
#[rustfmt::skip]
const INVERSE: [(f64, f64, f64, f64, f64, f64); 18] = [
    (6378137.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (7378137.0, 0.0, 0.0, 0.0, 0.0, 1000000.000000000000000000),
    (0.0, 0.0, 6356752.0, 1.570796326794896619231322, 0.0, -0.3142451794975639665996337),
    (0.0, 0.0, -7000000.0, -1.570796326794896619231322, 0.0, 643247.6857548205024360334),
    (1.0, 0.0, 6400000.0, 1.570796171583872220260194, 0.0, 43247.68575489810794823289),
    (-3.0, 2.0, -6356000.0, -1.570795763325496467642678, 2.553590050042225687217032, -752.3142441636886567658286),
    (4510731.0, 4510731.0, 0.0, 0.0, 0.7853981633974483096156608, 999.9564167534025762901806),
    (3194419.0, -3194419.0, 4487348.0, 0.7853981407741308284314932, -0.7853981633974483096156608, -0.4341724368678939145655526),
    (4000000.0, 3000000.0, 5000000.0, 0.7884223904025660190009920, 0.6435011087932843868028092, 703646.5135481512635165801),
    (-2500000.0, 1200000.0, -5800000.0, -1.127387872173759310519256, 2.694072678432623358741001, 68126.09966305271699264734),
    (42164000.0, 0.0, 0.0, 0.0, 0.0, 35785863.00000000000000000),
    (29814000.0, 29814000.0, 1000000.0, 0.02373684948644045929467502, 0.7853981633974483096156608, 35797095.13230909311966171),
    (-1000000.0, -6000000.0, 2900000.0, 0.4473537156248137459388600, -1.735945004209523457510450, 364532.8180350397205364844),
    (4400000.0, 0.0, 4500000.0, 0.8000301801211304098629546, 0.0, -73529.48295547533194889758),
    (100000.0, 0.0, 50000.0, 0.6487964810445284914378524, 0.0, -6260445.406241326226830986),
    (600000.0, -800000.0, -300000.0, -0.3031114102825667621670386, -0.9272952180016122324285125, -5332274.852579547705341099),
    (1.0, 1.0, 100000000.0, 1.570796312658817077372670, 0.7853981633974483096156608, 93643247.68575483049815374),
    (6378137.0, 1.0, 1.0, 1.578422502906813593400855e-7, 1.567855942887385150385566e-7, 1.573139222897102649204346e-7),
];

/// `(φ, λ, h, x, y, z)`: dyadic angles and integer heights (exact doubles) and the forward map
/// `((N + h) cos φ cos λ, (N + h) cos φ sin λ, (N(1 − e²) + h) sin φ)` at 80 digits.
#[rustfmt::skip]
const FORWARD: [(f64, f64, f64, f64, f64, f64); 8] = [
    (0.0, 0.0, 0.0, 6378137.000000000000000000, 0.0, 0.0),
    (0.5000000000000000000000000, 0.2500000000000000000000000, 0.0, 5427511.257995918760525721, 1385871.152065481331351040, 3039710.906851825772407514),
    (-0.6250000000000000000000000, 3.000000000000000000000000, 550000.0, -5568114.543700982492222597, 793715.4796461848079106730, -4032906.665752215426339869),
    (0.7500000000000000000000000, -2.000000000000000000000000, 700000.0, -2158249.396627034153402634, -4715860.966490190574567887, 4802359.990504663499617467),
    (1.562500000000000000000000, 1.000000000000000000000000, 35786000.0, 189095.7809856816534154751, 294499.2300067521891813332, 42141300.52572950084753804),
    (-0.0009765625000000000000000000, 0.0009765625000000000000000000, -1000.0, 6377130.938649975344062199, 6227.668911998123232410569, -6185.974942074592818115593),
    (1.570312500000000000000000, -1.500000000000000000000000, 400000.0, 232.7130469983141939196824, -3281.584402909137591440041, 6756751.518392310286441565),
    (-1.570312500000000000000000, 2.500000000000000000000000, 0.0, -2480.576770291683707031007, 1853.046157420150067783990, -6356751.565209982865139226),
];

/// (G2) Vermeille's closed form returns the geodetic coordinates that solve the equation defining
/// them, solved independently at 80 digits by bracketed root finding — not by Vermeille.
///
/// The latitude and longitude bounds are 4 ulps of `π/2`, 4.4e-16 rad, which is 2.8 nm on the
/// surface. The height bound is 4 ulps of the larger of `|r|` and `a`: the height is a difference of
/// two numbers of that size, and the last bit of either is the floor on it (1.5e-8 m at GEO). The
/// inward normal is checked at the same points against `−(cos φ cos λ, cos φ sin λ, sin φ)` at the
/// reference latitude, to 4 ulps of unity per component.
#[test]
fn the_inverse_solves_the_defining_equation_of_the_geodetic_latitude() {
    let angle_tol = 4.0 * f64::EPSILON * core::f64::consts::FRAC_PI_2;
    let (mut worst_lat, mut worst_lon, mut worst_h, mut worst_n) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (x, y, z, lat, lon, h) in INVERSE {
        let p = Position::<Itrf>::new(Vec3::new(x, y, z));
        let g = itrf_to_geodetic(p).unwrap();
        let r = (x * x + y * y + z * z).sqrt();
        let h_tol = 4.0 * f64::EPSILON * r.max(EQUATORIAL_RADIUS);
        let (dlat, dlon, dh) = (
            (g.latitude.radians() - lat).abs(),
            (g.longitude.radians() - lon).abs(),
            (g.height.metres() - h).abs(),
        );
        assert!(
            dlat <= angle_tol,
            "({x}, {y}, {z}): latitude off by {dlat:.3e} rad"
        );
        assert!(
            dlon <= angle_tol,
            "({x}, {y}, {z}): longitude off by {dlon:.3e} rad"
        );
        assert!(
            dh <= h_tol,
            "({x}, {y}, {z}): height off by {dh:.3e} m (allowed {h_tol:.3e})"
        );
        let down = geodetic_nadir(p).unwrap();
        let want = [-lat.cos() * lon.cos(), -lat.cos() * lon.sin(), -lat.sin()];
        let dn = [down.x - want[0], down.y - want[1], down.z - want[2]]
            .iter()
            .fold(0.0_f64, |m, d| m.max(d.abs()));
        assert!(
            dn <= 4.0 * f64::EPSILON,
            "({x}, {y}, {z}): normal off by {dn:.3e}"
        );
        worst_lat = worst_lat.max(dlat / angle_tol);
        worst_lon = worst_lon.max(dlon / angle_tol);
        worst_h = worst_h.max(dh / h_tol);
        worst_n = worst_n.max(dn / (4.0 * f64::EPSILON));
    }
    println!(
        "Vermeille vs the defining equation, fraction of bound: lat {worst_lat:.3}, lon {worst_lon:.3}, \
         h {worst_h:.3}, normal {worst_n:.3}"
    );
}

/// (G1) The forward map against the ellipsoid parametrisation evaluated at 80 digits on the same
/// doubles. Each coordinate to 4 ulps of `a + h`, the size of the terms it is built from.
#[test]
fn the_forward_map_is_the_ellipsoid_parametrisation_at_80_digits() {
    let mut worst = 0.0_f64;
    for (lat, lon, h, x, y, z) in FORWARD {
        let got = geodetic_to_itrf(Geodetic {
            latitude: Angle::from_radians(lat),
            longitude: Angle::from_radians(lon),
            height: Length::from_metres(h),
        })
        .vector;
        let tol = 4.0 * f64::EPSILON * (EQUATORIAL_RADIUS + h.abs());
        for (g, w) in [(got.x, x), (got.y, y), (got.z, z)] {
            let d = (g - w).abs();
            assert!(
                d <= tol,
                "({lat}, {lon}, {h}): {g} against {w}, off by {d:.3e} m"
            );
            worst = worst.max(d / tol);
        }
    }
    println!("forward map vs 80 digits: worst {worst:.3} of the bound");
}
