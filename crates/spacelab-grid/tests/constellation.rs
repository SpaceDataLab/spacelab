//! The project's thesis, made a number: a constellation closes the revisit gap a single satellite
//! leaves. One 700 km satellite over a mid-latitude cell waits many hours between looks; six planes
//! spread in right ascension cut that to a fraction — the union of their access windows.
//!
//! Because J₂ is axisymmetric about the pole, rotating the base trajectory about `z` by a right
//! ascension gives a valid orbit in another plane — so six satellites cost one integration and six
//! rotations, not six propagations.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use spacelab_core::{Angle, Length, Mat3, Vec3};
use spacelab_coverage::{Window, access_windows, combined_metrics, metrics};
use spacelab_earth::{Geodetic, geodetic_to_itrf};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::forces::{J2, Newtonian, Plain, PointMass};
use spacelab_propagation::{Dop853, Tolerance};
use spacelab_sensors::{ElevationMask, OrbitBound};

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;
const J2_COEFF: f64 = 1.082_626_683e-3;
const OMEGA_EARTH: f64 = 7.292_115_146_706_979e-5;

#[test]
fn a_six_plane_constellation_beats_a_single_satellite_on_revisit() {
    // One 700 km, 51.6° orbit, propagated for a day.
    let a = RE + 700e3;
    let inc = 51.6_f64.to_radians();
    let v = (MU / a).sqrt();
    let y0 = [a, 0.0, 0.0, 0.0, v * inc.cos(), v * inc.sin()];
    let span = 86_400.0;
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

    // Ground cell: eastern Mediterranean.
    let g = Geodetic {
        latitude: Angle::from_degrees(35.0),
        longitude: Angle::from_degrees(33.0),
        height: Length::from_metres(0.0),
    };
    let cell = geodetic_to_itrf(g).vector;
    let zenith = {
        let (sla, cla) = (g.latitude.radians().sin(), g.latitude.radians().cos());
        let (slo, clo) = (g.longitude.radians().sin(), g.longitude.radians().cos());
        Vec3::new(cla * clo, cla * slo, sla)
    };
    let mask = ElevationMask::new(10.0_f64.to_radians()).unwrap();
    let orbit = OrbitBound {
        semi_major_axis: a,
        eccentricity: 0.0,
        mu: MU,
    };

    // Satellite k sits in a plane rotated by k·60° in right ascension. Its Earth-fixed position is
    // rot3(ω⊕·t) · rot3(−RAAN_k) · r_inertial(t) = rot3(ω⊕·t − RAAN_k) · r_inertial(t).
    let windows_of = |raan: f64| {
        let sat_at = |t: f64| {
            let s = traj.eval(t);
            Mat3::rot3(OMEGA_EARTH * t - raan) * Vec3::new(s[0], s[1], s[2])
        };
        let mut ws: Vec<Window> = Vec::new();
        access_windows(
            &mask,
            &orbit,
            OMEGA_EARTH,
            cell,
            zenith,
            sat_at,
            0.0,
            span,
            1e-2,
            |w| ws.push(w),
        );
        ws
    };

    // One satellite.
    let mut solo = windows_of(0.0);
    let solo_gap = metrics(&solo, 0.0, span).max_gap;
    solo.sort_by(|x, y| x.rise.partial_cmp(&y.rise).unwrap());

    // Six planes, unioned.
    let planes = 6;
    let mut all: Vec<Window> = Vec::new();
    for k in 0..planes {
        all.extend(windows_of(
            core::f64::consts::TAU * f64::from(k) / f64::from(planes),
        ));
    }
    let combined = combined_metrics(&mut all, 0.0, span);

    println!(
        "revisit max gap: 1 satellite {:.1} h → {planes} planes {:.1} h ({} combined passes)",
        solo_gap / 3600.0,
        combined.max_gap / 3600.0,
        combined.passes
    );

    // The constellation must materially cut the worst-case wait.
    assert!(
        combined.max_gap < 0.5 * solo_gap,
        "six planes should more than halve the {:.1} h gap",
        solo_gap / 3600.0
    );
    assert!(
        combined.passes > planes,
        "the union should hold many passes across the day"
    );
}
