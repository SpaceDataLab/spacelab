//! The inward geodetic normal — the "down" a nadir-pointing attitude holds, against the closed forms
//! it has to reproduce.
//!
//! `geodetic_nadir` reuses `itrf_to_geodetic`'s Vermeille solution rather than forming `cos φ` and
//! `sin φ` with trigonometry, which is faster and is exactly the kind of reuse that can be subtly
//! wrong. So nothing here is asserted against that route: the oracle is the definition — the outward
//! normal at geodetic latitude `φ` and longitude `λ` is `(cos φ cos λ, cos φ sin λ, sin φ)` — and the
//! deflection from `−r/|r|` is checked against `derivations/nadir_deflection.mac`'s closed form.

#![allow(clippy::unwrap_used)]

use spacelab_core::{Angle, Length, Vec3};
use spacelab_earth::{
    EQUATORIAL_RADIUS, FLATTENING, Geodetic, GeodeticError, geodetic_nadir, geodetic_to_itrf,
};
use spacelab_frames::{Itrf, Position};

/// `e²` on WGS-84, from the flattening the crate publishes.
fn e2() -> f64 {
    FLATTENING * (2.0 - FLATTENING)
}

fn at(lat_deg: f64, lon_deg: f64, height_m: f64) -> Position<Itrf> {
    geodetic_to_itrf(Geodetic {
        latitude: Angle::from_degrees(lat_deg),
        longitude: Angle::from_degrees(lon_deg),
        height: Length::from_metres(height_m),
    })
}

fn angle_between(u: Vec3, v: Vec3) -> f64 {
    u.cross(v).norm().atan2(u.dot(v))
}

/// The definition, at points spread over both hemispheres and four decades of height.
///
/// The bound is 1e-15 rad — about 0.2 nanoarcseconds — which is round-off on a unit vector and not a
/// tolerance chosen to pass. A normal built from the wrong latitude (geocentric rather than geodetic)
/// misses by up to 0.19°, twelve orders above this.
#[test]
fn the_normal_is_the_outward_ellipsoid_normal_negated() {
    for (lat, lon, h) in [
        (0.0_f64, 0.0_f64, 0.0_f64),
        (45.0, 0.0, 0.0),
        (45.0962, 137.0, 550_000.0),
        (-33.8688, 151.2093, 58.0),
        (89.9, 12.0, 100.0),
        (-89.99, -3.0, 800_000.0),
        (23.4, -179.0, 35_786_000.0),
    ] {
        let (phi, lam) = (lat.to_radians(), lon.to_radians());
        let outward = Vec3::new(phi.cos() * lam.cos(), phi.cos() * lam.sin(), phi.sin());
        let got = geodetic_nadir(at(lat, lon, h)).unwrap();
        assert!(
            (got.norm() - 1.0).abs() < 1e-15,
            "{lat},{lon},{h}: norm {}",
            got.norm()
        );
        assert!(
            angle_between(got, -outward) < 1e-15,
            "{lat},{lon},{h}: {}rad from the negated outward normal",
            angle_between(got, -outward)
        );
    }
}

/// The normal belongs to the *point on the ellipsoid*, not to the position, so climbing along it
/// changes nothing. This is the property that makes the direction a per-satellite hoist rather than
/// something that has to be recomputed as an orbit's radius varies within a pass.
#[test]
fn the_normal_does_not_depend_on_height() {
    for lat in [-72.0_f64, -17.5, 0.0, 33.0, 45.0962, 81.0] {
        let ground = geodetic_nadir(at(lat, 61.0, 0.0)).unwrap();
        for h in [1.0_f64, 550_000.0, 20_200_000.0, 3.5786e7] {
            let up = geodetic_nadir(at(lat, 61.0, h)).unwrap();
            assert!(
                angle_between(ground, up) < 1e-15,
                "latitude {lat}: the normal moved {} rad between 0 and {h} m",
                angle_between(ground, up)
            );
        }
    }
}

/// The deflection from the geocentric nadir, against the closed form
/// `atan(e²/(2√(1−e²)))` = 0.192 424 3°, and at the geodetic latitude `atan(1/√(1−e²))` where
/// `derivations/nadir_deflection.mac` puts the maximum — 45.096 2°, not 45°.
///
/// The two are asserted separately: the *value* to 1e-12 relative, and the *location* by scanning a
/// degree either side at a hundredth of a degree and finding nothing larger. A closed form that is
/// right at one point and wrong about where the maximum sits is the failure this catches.
#[test]
fn the_surface_deflection_peaks_at_the_published_angle_and_latitude() {
    let e2 = e2();
    let closed_form = (e2 / (2.0 * (1.0 - e2).sqrt())).atan();
    let peak_latitude = (1.0 / (1.0 - e2).sqrt()).atan().to_degrees();

    // The derivation's §1 prints both: `atan(e²/(2√(1−e²)))` exactly, and a 0.001°-resolution sweep
    // that lands 1.7e-9° below it. This is the closed form, which is the one with digits to assert.
    assert!(
        (closed_form.to_degrees() - 0.192_424_301_159_560).abs() < 1e-14,
        "the closed form is {}°, not the derivation's",
        closed_form.to_degrees()
    );
    assert!(
        (peak_latitude - 45.096_23).abs() < 1e-4,
        "the peak latitude is {peak_latitude}°"
    );

    let deflection = |lat: f64| {
        let p = at(lat, 25.0, 0.0);
        angle_between(
            geodetic_nadir(p).unwrap(),
            -p.vector.scale(1.0 / p.vector.norm()),
        )
    };

    let at_peak = deflection(peak_latitude);
    assert!(
        (at_peak / closed_form - 1.0).abs() < 1e-12,
        "deflection at the peak is {}° against the closed form {}°",
        at_peak.to_degrees(),
        closed_form.to_degrees()
    );

    for step in -100..=100 {
        let lat = peak_latitude + f64::from(step) / 100.0;
        assert!(
            deflection(lat) <= at_peak + 1e-18,
            "latitude {lat}° deflects {}° , above the claimed maximum {}°",
            deflection(lat).to_degrees(),
            at_peak.to_degrees()
        );
    }
}

/// The deflection falls with height, because `(N(1−e²)+h)/(N+h) → 1`, and the doc comment states
/// four values. They are the claim, so they are the test — 1e-6 ° is the width they are quoted to.
#[test]
fn the_deflection_falls_with_height_by_the_published_amounts() {
    let peak_latitude = (1.0 / (1.0 - e2()).sqrt()).atan().to_degrees();
    let mut previous = f64::INFINITY;
    for (h, want_deg) in [
        (0.0_f64, 0.192_424_3_f64),
        (550_000.0, 0.177_124_9),
        (700_000.0, 0.173_365_6),
        (3.5786e7, 0.029_066_5),
    ] {
        // The maximum over latitude moves with height, so scan rather than assume it stays put.
        let mut worst = 0.0_f64;
        for step in -400..=400 {
            let lat = peak_latitude + f64::from(step) / 100.0;
            let p = at(lat, -8.0, h);
            worst = worst.max(angle_between(
                geodetic_nadir(p).unwrap(),
                -p.vector.scale(1.0 / p.vector.norm()),
            ));
        }
        assert!(
            (worst.to_degrees() - want_deg).abs() < 1e-6,
            "at {h} m the worst deflection is {}°, not the documented {want_deg}°",
            worst.to_degrees()
        );
        assert!(
            worst < previous,
            "the deflection did not fall between the previous height and {h} m"
        );
        previous = worst;
    }
}

/// On the polar axis the meridian is undefined and the normal is not, so the axis itself is the
/// answer — including for `-0.0`, which a naive comparison reads as the northern hemisphere.
#[test]
fn the_polar_axis_has_a_normal_even_though_it_has_no_longitude() {
    let polar = EQUATORIAL_RADIUS * (1.0 - FLATTENING);
    for (z, want) in [
        (polar + 800_000.0, Vec3::new(0.0, 0.0, -1.0)),
        (-(polar + 800_000.0), Vec3::new(0.0, 0.0, 1.0)),
    ] {
        let got = geodetic_nadir(Position::new(Vec3::new(0.0, 0.0, z))).unwrap();
        assert_eq!(got, want, "on the axis at z = {z}");
    }
    // `-0.0` is not a northern hemisphere: `copysign` must read the sign bit, not the value.
    let got = geodetic_nadir(Position::new(Vec3::new(0.0, 0.0, -0.0))).unwrap_err();
    assert_eq!(got, GeodeticError::NearGeocentre, "the origin is refused");
}

/// The same 43 km lens `itrf_to_geodetic` refuses, refused here too rather than returning a
/// direction the ellipsoid does not define.
#[test]
fn deep_inside_the_earth_there_is_no_normal() {
    for p in [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1000.0, -2000.0, 500.0),
        Vec3::new(0.0, 0.0, 6_300_000.0 - 6_300_000.0),
    ] {
        assert_eq!(
            geodetic_nadir(Position::new(p)).unwrap_err(),
            GeodeticError::NearGeocentre,
            "{p:?} should have no geodetic normal"
        );
    }
}
