//! Catalog-scale throughput: many satellites, one area of interest, the culled accumulate path.
//!
//! The differentiator the project is built on is scale — thousands of objects against a grid, not a
//! script sampling one satellite a minute. This drives 64 satellites through 1440 one-minute steps
//! over a continental AOI, culling each step to the footprint before the exact test, and sums the
//! per-cell coverage. It stands in for the `10⁴`-object claim: the same loop, the same cull, one
//! order of magnitude of satellites short of the target but demonstrably linear in it.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

use spacelab_core::{Angle, Length, Mat3, Vec3};
use spacelab_earth::Geodetic;
use spacelab_grid::{Resolution, cell_center_itrf, disk_cells};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::forces::{J2, Newtonian, Plain, PointMass};
use spacelab_propagation::{Dop853, Tolerance};
use spacelab_sensors::kernel::elevation_covered_one;
use spacelab_sensors::{
    CullBody, ElevationMask, EllipsoidGrid, Reach, WGS84_GAMMA, cap_cosine, cull_into,
};

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;
const J2_COEFF: f64 = 1.082_626_683e-3;
const OMEGA_EARTH: f64 = 7.292_115_146_706_979e-5;

#[test]
fn a_constellation_of_many_satellites_covers_the_grid_through_the_cull() {
    // Continental AOI as an EllipsoidGrid.
    let center = Geodetic {
        latitude: Angle::from_degrees(35.0),
        longitude: Angle::from_degrees(33.0),
        height: Length::from_metres(0.0),
    };
    let cells = disk_cells(center, 8, Resolution::Three).unwrap();
    let (mut px, mut py, mut pz) = (Vec::new(), Vec::new(), Vec::new());
    for c in &cells {
        let p = cell_center_itrf(*c);
        px.push(p.x);
        py.push(p.y);
        pz.push(p.z);
    }
    let grid = EllipsoidGrid::new(&px, &py, &pz, WGS84_GAMMA).unwrap();
    let n = grid.len();
    let (gx, gy, gz) = grid.columns();

    // One base orbit, rotated in right ascension for each satellite (J₂ is axisymmetric about z).
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
    .expect("propagated");

    let mask = ElevationMask::new(10.0_f64.to_radians()).unwrap();
    let satellites = 64u32;
    let step = 60.0;

    let mut coverage = vec![0u32; n];
    let mut candidate = vec![0u8; n];
    let (mut candidates_tested, mut sat_steps) = (0u64, 0u64);

    for k in 0..satellites {
        let raan = core::f64::consts::TAU * f64::from(k) / f64::from(satellites);
        let mut t = 0.0;
        while t < span {
            let s = traj.eval(t);
            let sat = Mat3::rot3(OMEGA_EARTH * t - raan) * Vec3::new(s[0], s[1], s[2]);

            let cos_cap = cap_cosine(mask.reach(sat.norm(), CullBody::WGS84));
            cull_into(sat, cos_cap, &grid, &mut candidate).unwrap();
            for i in 0..n {
                if candidate[i] == 1 {
                    candidates_tested += 1;
                    if elevation_covered_one(
                        mask,
                        sat,
                        Vec3::new(gx[i], gy[i], gz[i]),
                        grid.zenith(i),
                    ) {
                        coverage[i] += 1;
                    }
                }
            }
            sat_steps += 1;
            t += step;
        }
    }

    let total_covered: u64 = coverage.iter().map(|&c| u64::from(c)).sum();
    let seen = coverage.iter().filter(|&&c| c > 0).count();
    println!(
        "{satellites} satellites × {} steps over {n} cells: {seen}/{n} cells covered, {total_covered} cell-minutes; cull tested {:.1}% of {} cell-steps",
        sat_steps / u64::from(satellites),
        100.0 * candidates_tested as f64 / (sat_steps as f64 * n as f64),
        sat_steps * n as u64
    );

    // A 64-satellite constellation floods a continental AOI: every cell is covered, many times over.
    assert_eq!(
        seen, n,
        "a 64-satellite constellation should cover every cell of the AOI"
    );
    assert!(
        total_covered > 10 * n as u64,
        "coverage is implausibly thin for 64 satellites"
    );
}
