//! The whole chain, end to end, against an implementation that shares none of it.
//!
//! ```text
//! TLE ──SGP4──▶ TEME ──GMST82──▶ PEF ──polar motion──▶ ITRF ──Vermeille──▶ lat, lon, h
//! ```
//!
//! The oracle is Skyfield. It is of the Vallado lineage for SGP4 — as is the `sgp4`
//! crate we call — so that stage is a transcription check. But its frame code and
//! its geodetic solver are its own, and its geodetic solver is iterative where ours
//! is closed-form. Agreement across the whole chain is the first evidence in this
//! project that the *model*, not merely our arithmetic, is right.
//!
//! Skyfield had no polar-motion table loaded, so the fixtures were produced with
//! `xp = yp = 0` and we match that. The residual difference is the TIO locator,
//! `2e-11` rad in 2008, which is 0.14 mm at the surface.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_core::{Angle, Epoch, Tt, Ut1, UtcDateTime};
use spacelab_earth::itrf_to_geodetic;
use spacelab_frames::teme_to_itrf;
use spacelab_sgp4::Satellite;

const FIXTURES: &str = include_str!("fixtures/iss_groundtrack_skyfield.csv");

/// ISS, epoch 2008-264. The element set from the `sgp4` crate's own documentation,
/// which in turn is Vallado's.
const LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
const LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

/// Angular tolerance, degrees. `1e-7°` is 11 mm on the surface.
///
/// The two chains differ by: the TIO locator we apply and Skyfield does not
/// (0.14 mm); the 32 ns quantum of the UT1 column (15 nm of rotation); and the
/// rounding of two different geodetic solvers (sub-millimetre, measured separately
/// against ERFA). Nothing here should reach a centimetre.
const TOL_DEG: f64 = 1e-7;

/// Height tolerance, metres.
const TOL_H_M: f64 = 1e-3;

#[test]
fn the_iss_ground_track_agrees_with_skyfield() {
    let satellite = Satellite::from_tle(LINE1, LINE2).expect("the ISS element set parses");

    let mut checked = 0;
    let (mut worst_lat, mut worst_lon, mut worst_h) = (0.0_f64, 0.0_f64, 0.0_f64);

    for line in FIXTURES.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 10, "expected ten columns");

        let utc = UtcDateTime::new(
            f[0].parse().unwrap(),
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
            f[4].parse().unwrap(),
            f[5].parse().unwrap(),
        )
        .expect("a real calendar instant");

        let ut1 = Epoch::<Ut1>::from_ns_since_j2000(f[6].parse::<i128>().unwrap());
        // TT enters only through the TIO locator, whose whole contribution is 0.14 mm.
        let tt = Epoch::<Tt>::from_ns_since_j2000(ut1.ns_since_j2000());

        let teme = satellite
            .propagate_at(utc)
            .expect("the ISS has not decayed");
        let itrf = teme_to_itrf(
            ut1,
            tt,
            Angle::from_radians(0.0),
            Angle::from_radians(0.0),
            0.0,
        )
        .state(teme);
        let geo = itrf_to_geodetic(itrf.position).expect("well outside the singularity");

        let (lat, lon, h): (f64, f64, f64) = (
            f[7].parse().unwrap(),
            f[8].parse().unwrap(),
            f[9].parse().unwrap(),
        );

        let dlat = (geo.latitude.degrees() - lat).abs();
        let dlon = {
            let d = (geo.longitude.degrees() - lon).rem_euclid(360.0);
            d.min(360.0 - d)
        };
        let dh = (geo.height.metres() - h).abs();

        worst_lat = worst_lat.max(dlat);
        worst_lon = worst_lon.max(dlon);
        worst_h = worst_h.max(dh);

        assert!(dlat < TOL_DEG, "lat off by {dlat:.3e}° at {utc}");
        assert!(dlon < TOL_DEG, "lon off by {dlon:.3e}° at {utc}");
        assert!(dh < TOL_H_M, "height off by {dh:.3e} m at {utc}");
        checked += 1;
    }

    assert!(checked >= 40, "thin fixture set: {checked}");
    println!(
        "ground track: {checked} points; worst dlat {worst_lat:.2e}°, dlon {worst_lon:.2e}°, dh {worst_h:.2e} m"
    );
}

/// A ground track that never leaves the ellipsoid is a ground track computed with
/// the wrong `ω × r`, or with geocentric latitude, or with a stale GMST. The ISS
/// is at 350 km and crosses the equator twice an orbit.
#[test]
fn the_track_actually_moves() {
    let satellite = Satellite::from_tle(LINE1, LINE2).unwrap();
    let mut latitudes = Vec::new();

    for minute in 0..100 {
        let utc = UtcDateTime::new(2008, 9, 20, 12, 0, 0.0).unwrap();
        let ns = utc.naive_ns_since_j2000() + i128::from(minute) * 60_000_000_000;
        let label = UtcDateTime::from_naive_ns_since_j2000(ns, false);

        let ut1 = Epoch::<Ut1>::from_ns_since_j2000(ns);
        let teme = satellite.propagate_at(label).unwrap();
        let itrf = teme_to_itrf(
            ut1,
            Epoch::<Tt>::from_ns_since_j2000(ns),
            Angle::from_radians(0.0),
            Angle::from_radians(0.0),
            0.0,
        )
        .state(teme);
        let geo = itrf_to_geodetic(itrf.position).unwrap();

        assert!(
            (300_000.0..450_000.0).contains(&geo.height.metres()),
            "the ISS left its orbit: {} m",
            geo.height.metres()
        );
        latitudes.push(geo.latitude.degrees());
    }

    let peak = latitudes.iter().fold(0.0_f64, |m, &l| m.max(l.abs()));
    assert!(
        (peak - 51.6).abs() < 1.0,
        "peak latitude {peak:.2}° should approach the 51.6° inclination"
    );
}
