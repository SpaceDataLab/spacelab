//! The headline the whole system computes: **how many satellites** an area of interest needs to
//! meet a revisit requirement. For a growing constellation, the AOI-wide worst-case revisit — the
//! largest gap any cell suffers — is measured from the union of every satellite's access windows.
//! This is catalog-scale coverage intelligence in miniature: the answer a mission is sized against.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use spacelab_core::{Angle, Length, Mat3, Vec3};
use spacelab_coverage::{Window, access_windows, combined_metrics};
use spacelab_earth::Geodetic;
use spacelab_grid::{Resolution, cell_center, cell_center_itrf, disk_cells};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::forces::{J2, Newtonian, Plain, PointMass};
use spacelab_propagation::{Dop853, Tolerance};
use spacelab_sensors::{ElevationMask, OrbitBound};

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;
const J2_COEFF: f64 = 1.082_626_683e-3;
const OMEGA_EARTH: f64 = 7.292_115_146_706_979e-5;

fn cell_zenith(g: Geodetic) -> Vec3 {
    let (sla, cla) = (g.latitude.radians().sin(), g.latitude.radians().cos());
    let (slo, clo) = (g.longitude.radians().sin(), g.longitude.radians().cos());
    Vec3::new(cla * clo, cla * slo, sla)
}

#[test]
fn the_aoi_wide_revisit_falls_as_planes_are_added() {
    // Base orbit: 700 km, 51.6°, one integration over a day.
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

    // AOI: a small H3 disk over the eastern Mediterranean.
    let aoi_center = Geodetic {
        latitude: Angle::from_degrees(35.0),
        longitude: Angle::from_degrees(33.0),
        height: Length::from_metres(0.0),
    };
    let cells = disk_cells(aoi_center, 1, Resolution::Four).unwrap();
    let geometry: Vec<(Vec3, Vec3)> = cells
        .iter()
        .map(|c| (cell_center_itrf(*c), cell_zenith(cell_center(*c))))
        .collect();

    let mask = ElevationMask::new(10.0_f64.to_radians()).unwrap();
    let orbit = OrbitBound {
        semi_major_axis: a,
        eccentricity: 0.0,
        mu: MU,
    };

    // Satellite in the plane rotated by `raan`: Earth-fixed position rot3(ω⊕·t − raan)·r_inertial(t).
    let windows_for = |raan: f64, center: Vec3, zenith: Vec3| {
        let sat_at = |t: f64| {
            let s = traj.eval(t);
            Mat3::rot3(OMEGA_EARTH * t - raan) * Vec3::new(s[0], s[1], s[2])
        };
        let mut ws: Vec<Window> = Vec::new();
        access_windows(
            &mask,
            &orbit,
            OMEGA_EARTH,
            center,
            zenith,
            sat_at,
            0.0,
            span,
            1e-2,
            |w| ws.push(w),
        );
        ws
    };

    // AOI-wide worst revisit for a constellation of `planes` satellites spread in right ascension.
    let worst_revisit = |planes: u32| -> f64 {
        geometry
            .iter()
            .map(|&(center, zenith)| {
                let mut all = Vec::new();
                for k in 0..planes {
                    all.extend(windows_for(
                        core::f64::consts::TAU * f64::from(k) / f64::from(planes),
                        center,
                        zenith,
                    ));
                }
                combined_metrics(&mut all, 0.0, span).max_gap
            })
            .fold(0.0_f64, f64::max)
    };

    let (one, six, eighteen) = (worst_revisit(1), worst_revisit(6), worst_revisit(18));
    println!(
        "AOI-wide worst revisit: 1 plane {:.1} h, 6 planes {:.1} h, 18 planes {:.1} h",
        one / 3600.0,
        six / 3600.0,
        eighteen / 3600.0
    );

    // Adding planes monotonically shrinks the worst-case wait, and 18 meets a 2 h requirement.
    assert!(six < one, "6 planes must beat 1");
    assert!(eighteen < six, "18 planes must beat 6");
    assert!(
        eighteen < 2.0 * 3600.0,
        "18 planes should meet a 2 h revisit SLA, got {:.1} h",
        eighteen / 3600.0
    );
}
