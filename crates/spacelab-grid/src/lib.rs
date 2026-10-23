//! H3 geodesic grid: tessellate a region into equal-area cells, and place each cell centre on the
//! WGS-84 ellipsoid in the Earth-fixed frame.
//!
//! The coverage engine reasons over cells: "which cells does a satellite see, and when". H3 gives
//! those cells the property that makes the reasoning sound — they are (very nearly) **equal area**,
//! so counting covered cells *is* measuring covered area, with no projection distortion to correct
//! (`notes/06-coverage-source-material.md`). This crate is the bridge: an area of interest to a set
//! of H3 cells, and each cell to the Cartesian point the sensor geometry needs.

pub mod output;

pub use output::{CellCoverage, to_csv, to_geojson, to_svg};

// Re-exported so consumers need not depend on `h3o` directly.
pub use h3o::{CellIndex, Resolution};

use h3o::LatLng;

use spacelab_core::{Angle, Length, Vec3};
use spacelab_earth::{Geodetic, geodetic_to_itrf};

/// The H3 cells within `k` grid rings of the cell containing `center`, at `resolution` — a hexagonal
/// approximation to a circular area of interest (`1 + 3k(k+1)` cells).
///
/// # Errors
///
/// [`h3o::error::InvalidLatLng`] if the centre's latitude or longitude is out of range; a geodetic
/// latitude in `[-90°, 90°]` and longitude in `[-180°, 180°]` always succeeds.
pub fn disk_cells(
    center: Geodetic,
    k: u32,
    resolution: Resolution,
) -> Result<Vec<CellIndex>, h3o::error::InvalidLatLng> {
    let ll = LatLng::new(center.latitude.degrees(), center.longitude.degrees())?;
    Ok(ll.to_cell(resolution).grid_disk::<Vec<_>>(k))
}

/// The geodetic coordinates of a cell's centre, at zero height above the ellipsoid.
#[must_use]
pub fn cell_center(cell: CellIndex) -> Geodetic {
    let ll = LatLng::from(cell);
    Geodetic {
        latitude: Angle::from_radians(ll.lat_radians()),
        longitude: Angle::from_radians(ll.lng_radians()),
        height: Length::from_metres(0.0),
    }
}

/// A cell's centre as an Earth-fixed Cartesian position on the WGS-84 ellipsoid.
#[must_use]
pub fn cell_center_itrf(cell: CellIndex) -> Vec3 {
    geodetic_to_itrf(cell_center(cell)).vector
}
