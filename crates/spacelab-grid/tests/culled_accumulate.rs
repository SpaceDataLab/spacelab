//! The catalog-scale coverage path: cull, then test only what survives.
//!
//! Per timestep the sub-satellite footprint touches a tiny fraction of a large area of interest, so
//! testing every cell is waste. The conservative spherical-cap cull (`spacelab-sensors::cull`) marks
//! the candidates — and its contract is that it **never drops a covered cell** — so accumulating over
//! the candidates gives byte-for-byte the same coverage map as accumulating over the whole grid,
//! while touching a fraction of the cells. This test is the proof of both: identical result, far
//! fewer exact evaluations.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]
#![allow(clippy::many_single_char_names)]

use spacelab_core::{Angle, Length, Mat3, Vec3};
use spacelab_earth::Geodetic;
use spacelab_grid::{Resolution, cell_center_itrf, disk_cells};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::forces::{J2, Newtonian, Plain, PointMass};
use spacelab_propagation::{Dop853, Tolerance};
use spacelab_sensors::ellipsoid_grid::elevation_accumulate_into;
use spacelab_sensors::kernel::elevation_covered_one;
use spacelab_sensors::{
    CullBody, ElevationMask, EllipsoidGrid, Reach, WGS84_GAMMA, cap_cosine, cull_into,
};

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;
const J2_COEFF: f64 = 1.082_626_683e-3;
const OMEGA_EARTH: f64 = 7.292_115_146_706_979e-5;

#[test]
fn culling_gives_the_same_map_while_testing_far_fewer_cells() {
    // A continental AOI: a large H3 disk, so most cells are far from the footprint at any instant.
    let center = Geodetic {
        latitude: Angle::from_degrees(35.0),
        longitude: Angle::from_degrees(33.0),
        height: Length::from_metres(0.0),
    };
    let cells = disk_cells(center, 9, Resolution::Three).unwrap();
    let (mut px, mut py, mut pz) = (Vec::new(), Vec::new(), Vec::new());
    for c in &cells {
        let p = cell_center_itrf(*c);
        px.push(p.x);
        py.push(p.y);
        pz.push(p.z);
    }
    let grid = EllipsoidGrid::new(&px, &py, &pz, WGS84_GAMMA).unwrap();
    let n = grid.len();

    // A 700 km orbit sampled every minute for a day, carried into the Earth-fixed frame.
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

    let mut full = vec![0u32; n];
    let mut culled = vec![0u32; n];
    let mut candidate = vec![0u8; n];
    let (mut candidates_total, mut steps) = (0u64, 0u64);

    let (gx, gy, gz) = grid.columns();
    let mut t = 0.0;
    while t < span {
        let s = traj.eval(t);
        let sat = Mat3::rot3(OMEGA_EARTH * t) * Vec3::new(s[0], s[1], s[2]);

        // full: accumulate over every cell
        elevation_accumulate_into(mask, sat, &grid, &mut full).unwrap();

        // culled: mark candidates by the cap, then test only those
        let cos_cap = cap_cosine(mask.reach(sat.norm(), CullBody::WGS84));
        cull_into(sat, cos_cap, &grid, &mut candidate).unwrap();
        for i in 0..n {
            if candidate[i] == 1 {
                candidates_total += 1;
                let cell = Vec3::new(gx[i], gy[i], gz[i]);
                if elevation_covered_one(mask, sat, cell, grid.zenith(i)) {
                    culled[i] += 1;
                }
            }
        }
        steps += 1;
        t += 60.0;
    }

    let tested_fraction = candidates_total as f64 / (steps as f64 * n as f64);
    println!(
        "{n} cells × {steps} steps: cull tested {:.2}% of cell-steps; maps {}",
        100.0 * tested_fraction,
        if full == culled {
            "identical"
        } else {
            "DIFFER"
        }
    );

    // The cull must not change the answer — its whole contract.
    assert_eq!(full, culled, "culling dropped or added coverage");
    // And it must earn its place: only a small slice of the AOI is ever a candidate.
    assert!(
        tested_fraction < 0.2,
        "cull barely narrowed the search: {:.1}%",
        100.0 * tested_fraction
    );
    assert!(
        full.iter().any(|&c| c > 0),
        "the orbit covered nothing — check the geometry"
    );
}
