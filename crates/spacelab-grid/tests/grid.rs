//! H3 grid: round-trip, equal-area, and the bridge into the coverage engine.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use h3o::Resolution;

use spacelab_core::{Angle, Length};
use spacelab_earth::{Geodetic, geodetic_to_itrf};
use spacelab_grid::{cell_center, cell_center_itrf, disk_cells};
use spacelab_sensors::{
    ElevationMask, EllipsoidGrid, WGS84_GAMMA, ellipsoid_grid::elevation_covered_into,
};

fn geodetic(lat_deg: f64, lon_deg: f64, height_m: f64) -> Geodetic {
    Geodetic {
        latitude: Angle::from_degrees(lat_deg),
        longitude: Angle::from_degrees(lon_deg),
        height: Length::from_metres(height_m),
    }
}

#[test]
fn a_point_lands_in_a_cell_whose_centre_is_within_a_cell_of_it() {
    // point → cell → centre: the centre is at most a cell radius away (res 8 ≈ 0.5 km ≈ 0.005°).
    let point = geodetic(35.31, 33.17, 0.0);
    let cell = disk_cells(point, 0, Resolution::Eight).unwrap()[0];
    let back = cell_center(cell);
    assert!(
        (back.latitude.degrees() - 35.31).abs() < 0.01,
        "lat {}",
        back.latitude.degrees()
    );
    assert!(
        (back.longitude.degrees() - 33.17).abs() < 0.01,
        "lon {}",
        back.longitude.degrees()
    );
}

#[test]
fn a_disk_of_k_rings_has_the_h3_cell_count() {
    // |disk(k)| = 1 + 3k(k+1): 1, 7, 19, 37 for k = 0, 1, 2, 3.
    for (k, expected) in [(0u32, 1usize), (1, 7), (2, 19), (3, 37)] {
        let cells = disk_cells(geodetic(35.0, 33.0, 0.0), k, Resolution::Five).unwrap();
        assert_eq!(
            cells.len(),
            expected,
            "disk({k}) should have {expected} cells"
        );
    }
}

#[test]
fn cell_centres_lie_on_the_ellipsoid() {
    // Every cell centre (height 0) sits on the WGS-84 surface: 6357–6378 km from geocentre.
    for cell in disk_cells(geodetic(35.0, 33.0, 0.0), 3, Resolution::Five).unwrap() {
        let r = cell_center_itrf(cell).norm();
        assert!(
            (6_356_000.0..6_379_000.0).contains(&r),
            "cell radius {r:.0} m off the ellipsoid"
        );
    }
}

#[test]
fn h3_cells_are_nearly_equal_area() {
    // The property the coverage engine relies on: counting cells ≈ measuring area.
    let cells = disk_cells(geodetic(35.0, 33.0, 0.0), 4, Resolution::Five).unwrap();
    let areas: Vec<f64> = cells.iter().map(|c| c.area_km2()).collect();
    let min = areas.iter().copied().fold(f64::INFINITY, f64::min);
    let max = areas.iter().copied().fold(0.0_f64, f64::max);
    assert!(
        max / min < 1.15,
        "cell areas vary by {:.1}% across the disk",
        100.0 * (max / min - 1.0)
    );
}

#[test]
fn an_overhead_satellite_covers_the_whole_small_aoi() {
    // The bridge into the engine: an AOI of H3 cells becomes an EllipsoidGrid, and the elevation
    // kernel marks coverage. A satellite directly over the AOI centre sees all of a small disk.
    let center = geodetic(35.0, 33.0, 0.0);
    let cells = disk_cells(center, 4, Resolution::Five).unwrap(); // ~70 km AOI

    let mut px = Vec::new();
    let mut py = Vec::new();
    let mut pz = Vec::new();
    for cell in &cells {
        let p = cell_center_itrf(*cell);
        px.push(p.x);
        py.push(p.y);
        pz.push(p.z);
    }
    let grid = EllipsoidGrid::new(&px, &py, &pz, WGS84_GAMMA).unwrap();

    // satellite 700 km directly above the AOI centre
    let sat = geodetic_to_itrf(geodetic(35.0, 33.0, 700e3)).vector;
    let mask = ElevationMask::new(10.0_f64.to_radians()).unwrap();

    let mut covered = vec![0u8; cells.len()];
    elevation_covered_into(mask, sat, &grid, &mut covered).unwrap();
    let n_covered: usize = covered.iter().map(|&c| c as usize).sum();
    assert_eq!(
        n_covered,
        cells.len(),
        "overhead, every cell of a 70 km AOI should be covered"
    );

    // and a satellite over the antipode sees none of them
    let far = geodetic_to_itrf(geodetic(-35.0, -147.0, 700e3)).vector;
    let mut none = vec![0u8; cells.len()];
    elevation_covered_into(mask, far, &grid, &mut none).unwrap();
    assert_eq!(
        none.iter().map(|&c| c as usize).sum::<usize>(),
        0,
        "the far side sees nothing"
    );
}
