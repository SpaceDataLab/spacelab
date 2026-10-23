//! End-to-end coverage: a numerically-propagated orbit, a rotating Earth, a fixed ground station,
//! and the access-window scanner — the whole pipeline the separate pieces were built for.
//!
//! The satellite is integrated under point-mass + J₂ gravity in an inertial frame; the ground station
//! is fixed in the Earth-fixed frame; the transform between them is the Earth-rotation angle
//! `ω⊕·t`. Feeding the Earth-fixed satellite position into `access_windows` over 24 h yields the
//! passes over the station. There is no synthetic orbit here — the trajectory is the real integrator's
//! dense output, sampled by the adaptive scanner.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_core::{Mat3, Vec3};
use spacelab_coverage::{Window, access_windows, metrics};
use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::{J2, Newtonian, Plain, PointMass};
use spacelab_propagation::{Dop853, Tolerance};
use spacelab_sensors::{ElevationMask, OrbitBound};

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;
const J2_COEFF: f64 = 1.082_626_683e-3;
const OMEGA_EARTH: f64 = 7.292_115_146_706_979e-5;

/// A ground station on a spherical Earth at geodetic-ish `(lat, lon)`; on a sphere the outward
/// normal is radial, so `cell` and `zenith` share a direction.
fn station(lat: f64, lon: f64) -> (Vec3, Vec3) {
    let (sla, cla) = (lat.sin(), lat.cos());
    let (slo, clo) = (lon.sin(), lon.cos());
    let up = Vec3::new(cla * clo, cla * slo, sla);
    (up.scale(RE), up)
}

#[test]
#[allow(clippy::too_many_lines)] // one end-to-end integration scenario
fn a_propagated_orbit_makes_physical_passes_over_a_ground_station() {
    // 700 km, 51.6° circular orbit — its ground track reaches the station's latitude.
    let a = RE + 700e3;
    let inc = 51.6_f64.to_radians();
    let speed = (MU / a).sqrt();
    let y0 = [a, 0.0, 0.0, 0.0, speed * inc.cos(), speed * inc.sin()];

    let span = 86_400.0; // 24 hours
    let field = Newtonian((
        PointMass { mu: MU },
        J2 {
            mu: MU,
            re: RE,
            j2: J2_COEFF,
        },
    ));
    let traj = integrate(
        &Dop853,
        &Plain(field),
        0.0,
        &y0,
        span,
        60.0,
        Controls::new(Tolerance::new(1e-11, 1e-11), 1e-3),
    )
    .expect("propagated 24 h");

    // Inertial → Earth-fixed by the Earth-rotation angle ω⊕·t.
    let sat_at = |t: f64| {
        let s = traj.eval(t);
        Mat3::rot3(OMEGA_EARTH * t) * Vec3::new(s[0], s[1], s[2])
    };

    let (cell, zenith) = station(35.0_f64.to_radians(), 33.0_f64.to_radians()); // Cyprus-ish
    let mask = ElevationMask::new(10.0_f64.to_radians()).expect("valid mask");
    let orbit = OrbitBound {
        semi_major_axis: a,
        eccentricity: 0.0,
        mu: MU,
    };

    let mut passes: Vec<Window> = Vec::new();
    access_windows(
        &mask,
        &orbit,
        OMEGA_EARTH,
        cell,
        zenith,
        sat_at,
        0.0,
        span,
        1e-3,
        |w| passes.push(w),
    );

    // switching function in the Earth-fixed frame, for spot checks
    let g = |t: f64| {
        let s = sat_at(t);
        let los = Vec3::new(s.x - cell.x, s.y - cell.y, s.z - cell.z);
        los.dot(zenith) - 10.0_f64.to_radians().sin() * los.norm()
    };

    println!("passes over 24 h at 10° elevation: {}", passes.len());
    for (k, w) in passes.iter().enumerate() {
        println!(
            "  pass {k}: {:.0}–{:.0} s  ({:.1} min)",
            w.rise,
            w.set,
            w.duration() / 60.0
        );
    }

    let stats = metrics(&passes, 0.0, span);
    println!(
        "coverage {:.3}%, max gap {:.1} h, mean gap {:.1} h",
        100.0 * stats.coverage_fraction,
        stats.max_gap / 3600.0,
        stats.mean_gap / 3600.0
    );

    // A mid-latitude station under a 51.6° LEO sees a handful of passes a day, each a few minutes.
    assert!(
        (2..=12).contains(&passes.len()),
        "implausible pass count {}",
        passes.len()
    );
    // A single ground station sees the satellite only briefly, and there is a long daytime gap while
    // the orbit plane is turned away — several hours, the revisit latency this station suffers.
    assert!(
        stats.coverage_fraction < 0.05,
        "one station cannot cover 5% of the day: {}",
        stats.coverage_fraction
    );
    assert!(
        (2.0 * 3600.0..20.0 * 3600.0).contains(&stats.max_gap),
        "max gap {:.0} s implausible",
        stats.max_gap
    );
    let mut prev_set = f64::NEG_INFINITY;
    for w in &passes {
        assert!(w.rise > prev_set, "passes overlap or are out of order");
        prev_set = w.set;
        assert!(
            (60.0..900.0).contains(&w.duration()),
            "pass duration {:.0} s off scale",
            w.duration()
        );
        // the pass is real: covered at closest approach, not covered just before it opens
        assert!(
            g(w.midpoint()) > 0.0,
            "not actually covered at the culmination"
        );
        if w.rise > 1.0 {
            assert!(
                g(w.rise - 1.0) < 0.0,
                "already covered a second before the window opens"
            );
        }
    }
}
