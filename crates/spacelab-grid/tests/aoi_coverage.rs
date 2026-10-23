//! The whole MVP in one test: an area of interest tessellated into H3 cells, a satellite propagated
//! under J₂ over a rotating Earth, and the revisit map that results — for every cell, how often it
//! is seen and the worst wait between looks.
//!
//! This is the pipeline the separate crates exist to compose: `spacelab-grid` (the AOI cells),
//! `spacelab-propagation` (the orbit), the Earth-rotation transform, `spacelab-sensors` (the
//! elevation constraint), and `spacelab-coverage` (per-cell access windows and their gap metrics).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use h3o::Resolution;
use spacelab_testkit::nan_max;

use spacelab_core::{Angle, Length, Mat3, Vec3};
use spacelab_coverage::{Window, access_windows, metrics};
use spacelab_earth::Geodetic;
use spacelab_grid::{cell_center, cell_center_itrf, disk_cells};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::forces::{J2, Newtonian, Plain, PointMass};
use spacelab_propagation::{Dop853, Tolerance};
use spacelab_sensors::{ElevationMask, OrbitBound};

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;
const J2_COEFF: f64 = 1.082_626_683e-3;
const OMEGA_EARTH: f64 = 7.292_115_146_706_979e-5;

/// The outward geodetic normal (zenith) at a cell's centre.
fn cell_zenith(cell: h3o::CellIndex) -> Vec3 {
    let g = cell_center(cell);
    let (sla, cla) = (g.latitude.radians().sin(), g.latitude.radians().cos());
    let (slo, clo) = (g.longitude.radians().sin(), g.longitude.radians().cos());
    Vec3::new(cla * clo, cla * slo, sla)
}

#[test]
fn the_pipeline_produces_a_revisit_map_over_an_area_of_interest() {
    // AOI: a ~1000 km disk of H3 cells around the eastern Mediterranean, spanning ~26°–44° latitude.
    let aoi_center = Geodetic {
        latitude: Angle::from_degrees(35.0),
        longitude: Angle::from_degrees(33.0),
        height: Length::from_metres(0.0),
    };
    let cells = disk_cells(aoi_center, 3, Resolution::Two).unwrap();

    // Propagate a 700 km, 51.6° orbit for 24 h under point mass + J₂.
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

    let sat_at = |t: f64| {
        let s = traj.eval(t);
        Mat3::rot3(OMEGA_EARTH * t) * Vec3::new(s[0], s[1], s[2])
    };
    let mask = ElevationMask::new(10.0_f64.to_radians()).unwrap();
    let orbit = OrbitBound {
        semi_major_axis: a,
        eccentricity: 0.0,
        mu: MU,
    };

    // Per-cell revisit: run the access scanner over every cell, reduce its windows to metrics.
    let mut passes_per_cell = Vec::with_capacity(cells.len());
    let mut worst_gap = 0.0_f64;
    let mut best_gap = f64::INFINITY;
    let mut cells_seen = 0;
    for cell in &cells {
        let center = cell_center_itrf(*cell);
        let zenith = cell_zenith(*cell);
        let mut windows: Vec<Window> = Vec::new();
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
            |w| windows.push(w),
        );
        let m = metrics(&windows, 0.0, span);
        passes_per_cell.push(m.passes);
        if m.passes > 0 {
            cells_seen += 1;
            worst_gap = nan_max(worst_gap, m.max_gap);
            best_gap = best_gap.min(m.max_gap);
        }
    }

    let total: u32 = passes_per_cell.iter().sum();
    let mean = f64::from(total) / cells.len() as f64;
    println!(
        "AOI {} cells: {cells_seen} seen, mean {mean:.1} passes/cell, revisit gap {:.1}–{:.1} h",
        cells.len(),
        best_gap / 3600.0,
        worst_gap / 3600.0
    );

    // Every cell in this latitude band is overflown by a 51.6° orbit within a day.
    assert_eq!(
        cells_seen,
        cells.len(),
        "some AOI cell was never seen in 24 h"
    );
    // Coverage is not uniform: the revisit gap varies across the AOI (different latitudes, different
    // pass geometry), which is the whole reason to compute a map and not a single number.
    assert!(
        worst_gap > best_gap * 1.2,
        "revisit map is suspiciously flat: {best_gap:.0}–{worst_gap:.0} s"
    );
    // And no cell waits the whole day — this LEO revisits a mid-latitude band several times over.
    assert!(
        worst_gap < 20.0 * 3600.0,
        "a cell went uncovered for {:.1} h",
        worst_gap / 3600.0
    );
}
