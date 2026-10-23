//! Reality oracle — a real Sentinel-1A element set against Sentinel-1's *published* orbit, and the
//! SAR sensor exercised in the full coverage chain.
//!
//! Every other test compares us to another simulator (Orekit, Skyfield, pyerfa). This one compares
//! us to the world: the propagated orbit must match Sentinel-1's designed, published sun-synchronous
//! orbit — inclination 98.18°, mean altitude ~693 km, and a nodal precession locked to the Sun at
//! ~0.9856°/day — and the [`SideLookingSAR`] incidence band must restrict access to a strict subset
//! of geometric visibility over a real overpass.
//!
//! The full Phase-4 gate ("published *acquisitions* fall inside predicted windows") additionally
//! needs a historical acquisition paired with its contemporaneous element set (Space-Track,
//! authenticated); this validates the orbit and the SAR geometry, which is the part that needs no
//! credentials. The element set is a real one from CelesTrak, epoch 2026-07-11 (NORAD 39634).

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_possible_truncation
)]
#![allow(clippy::doc_markdown)] // SGP4, CelesTrak, Sentinel are names in prose

use spacelab::{EopSource, teme_to_itrf_at};
use spacelab_core::{Angle, Length, UtcDateTime, Vec3};
use spacelab_coverage::{Window, access_windows};
use spacelab_earth::{Geodetic, geodetic_to_itrf};
use spacelab_eop::EopSeries;
use spacelab_frames::EARTH_ROTATION_RATE;
use spacelab_sensors::{Constraint, ElevationMask, OrbitBound, SideLookingSAR};
use spacelab_sgp4::Satellite;

const MU: f64 = 3.986_004_418e14;
const EARTH_RADIUS: f64 = 6_378_137.0;

// Sentinel-1A, CelesTrak GP, epoch 2026-07-11 (26192.5).
const NAME: &str = "SENTINEL-1A";
const LINE1: &str = "1 39634U 14016A   26192.50461863  .00000208  00000+0  53758-4 0  9991";
const LINE2: &str = "2 39634  98.1786 199.6823 0001447  85.3804 274.7561 14.59200564653561";

/// The IERS series across the overpass, cut from `finals2000A.all` at MJD 61228..61238.
const EOP: &str = include_str!("fixtures/finals2000A_2026192.txt");

/// TEME position and velocity at a UTC instant, and the seconds-since-J2000 naive coordinate.
fn state_at(sat: &Satellite, utc: UtcDateTime) -> (Vec3, Vec3) {
    let sv = sat.propagate_at(utc).expect("Sentinel-1A has not decayed");
    (sv.position.vector, sv.velocity.vector)
}

/// Osculating semi-major axis, inclination and RAAN from a TEME state, in SI / radians.
fn elements(r: Vec3, v: Vec3) -> (f64, f64, f64) {
    let rn = r.norm();
    let energy = v.dot(v) / 2.0 - MU / rn;
    let a = -MU / (2.0 * energy);
    let h = r.cross(v);
    let inc = (h.z / h.norm()).acos();
    // Node line n = ẑ × h = (−h_y, h_x, 0); RAAN measured from x to the node.
    let raan = h.x.atan2(-h.y).rem_euclid(core::f64::consts::TAU);
    (a, inc, raan)
}

#[test]
fn sentinel1a_orbit_matches_the_published_sun_synchronous_design() {
    assert!(LINE1.contains("39634") && NAME == "SENTINEL-1A");
    let sat = Satellite::from_tle(LINE1, LINE2).expect("Sentinel-1A element set parses");

    let epoch = UtcDateTime::new(2026, 7, 11, 12, 0, 0.0).unwrap();
    let (r0, v0) = state_at(&sat, epoch);
    let (a, inc, raan0) = elements(r0, v0);
    let altitude = a - EARTH_RADIUS;

    // Published Sentinel-1: i = 98.18°, mean altitude 693 km.
    assert!(
        (inc.to_degrees() - 98.18).abs() < 0.15,
        "inclination {:.3}° is not Sentinel-1's 98.18°",
        inc.to_degrees()
    );
    assert!(
        (640e3..740e3).contains(&altitude),
        "altitude {:.0} km is not Sentinel-1's ~693 km",
        altitude / 1e3
    );

    // Sun-synchronous: the ascending node precesses eastward at the mean solar rate, ~0.9856°/day.
    // SGP4's J2 secular Ω̇ must reproduce it, which is the whole reason the inclination is 98.18°.
    let (r1, v1) = state_at(&sat, UtcDateTime::new(2026, 7, 12, 12, 0, 0.0).unwrap());
    let (_, _, raan1) = elements(r1, v1);
    let draan_deg = (raan1 - raan0)
        .rem_euclid(core::f64::consts::TAU)
        .to_degrees();
    assert!(
        (0.95..1.02).contains(&draan_deg),
        "nodal precession {draan_deg:.4}°/day is not sun-synchronous (~0.9856)"
    );
    println!(
        "Sentinel-1A: i = {:.3}°, altitude = {:.0} km, node precession = {draan_deg:.4}°/day",
        inc.to_degrees(),
        altitude / 1e3
    );
}

#[test]
#[allow(clippy::too_many_lines)] // one end-to-end integration scenario
fn sar_incidence_band_restricts_access_to_a_subset_of_visibility() {
    let sat = Satellite::from_tle(LINE1, LINE2).expect("parses");

    // A mid-latitude AOI. Sentinel-1 overflies everywhere within its 12-day repeat.
    let g = Geodetic {
        latitude: Angle::from_degrees(52.0),
        longitude: Angle::from_degrees(13.0),
        height: Length::from_metres(0.0),
    };
    let cell = geodetic_to_itrf(g).vector;
    let (sla, cla) = (g.latitude.radians().sin(), g.latitude.radians().cos());
    let (slo, clo) = (g.longitude.radians().sin(), g.longitude.radians().cos());
    let zenith = Vec3::new(cla * clo, cla * slo, sla);

    let series = EopSeries::parse_finals2000a(EOP).expect("the EOP excerpt parses");
    let start_ns = UtcDateTime::new(2026, 7, 11, 0, 0, 0.0)
        .unwrap()
        .naive_ns_since_j2000();
    let sat_at = |t: f64| {
        let ns = start_ns + (t * 1e9) as i128;
        let label = UtcDateTime::from_naive_ns_since_j2000(ns, false);
        let teme = sat.propagate_at(label).expect("propagates");
        let (transform, _) = teme_to_itrf_at(label, EopSource::Series(&series))
            .expect("the overpass lies inside the excerpt");
        transform.state(teme).position.vector
    };

    let orbit = OrbitBound {
        semi_major_axis: 7_073e3,
        eccentricity: 0.0002,
        mu: MU,
    };
    let span = 6.0 * 86_400.0; // six days: enough for several overpasses in the repeat cycle
    let horizon = ElevationMask::HORIZON;
    let sar = SideLookingSAR::new(29f64.to_radians(), 46f64.to_radians()).expect("IW band");

    let mut visible: Vec<Window> = Vec::new();
    let mut imaged: Vec<Window> = Vec::new();
    access_windows(
        &horizon,
        &orbit,
        EARTH_ROTATION_RATE,
        cell,
        zenith,
        sat_at,
        0.0,
        span,
        1e-3,
        |w| visible.push(w),
    );
    access_windows(
        &sar,
        &orbit,
        EARTH_ROTATION_RATE,
        cell,
        zenith,
        sat_at,
        0.0,
        span,
        1e-3,
        |w| imaged.push(w),
    );

    assert!(
        !visible.is_empty(),
        "Sentinel-1A should overfly a mid-latitude AOI within six days"
    );
    assert!(
        !imaged.is_empty(),
        "the SAR should have valid incidence on some overpass"
    );
    assert!(
        imaged.len() <= visible.len(),
        "SAR access ({}) cannot exceed geometric visibility ({})",
        imaged.len(),
        visible.len()
    );

    // Every SAR window sits inside a visibility window, and the incidence band makes it strictly
    // shorter than the full horizon-to-horizon pass — the constraint is doing real geometric work.
    let total_visible: f64 = visible.iter().map(|w| w.set - w.rise).sum();
    let total_imaged: f64 = imaged.iter().map(|w| w.set - w.rise).sum();
    assert!(
        total_imaged < total_visible,
        "SAR should image only part of each pass"
    );
    for iw in &imaged {
        let mid = f64::midpoint(iw.rise, iw.set);
        assert!(
            visible
                .iter()
                .any(|v| v.rise - 1.0 <= mid && mid <= v.set + 1.0),
            "a SAR window at t≈{mid:.0}s falls outside every visibility window"
        );
        // Confirm the incidence there really is inside the band, not merely bracketed.
        let p = sat_at(mid);
        let geom = spacelab_sensors::AccessGeometry {
            satellite: p,
            cell,
            zenith,
        };
        assert!(
            sar.covered(&geom),
            "SAR reported a window where the incidence is out of band"
        );
    }
    println!(
        "Sentinel-1A over 52°N,13°E in 6 days: {} visible passes ({:.0}s), {} SAR-valid ({:.0}s)",
        visible.len(),
        total_visible,
        imaged.len(),
        total_imaged
    );
}
