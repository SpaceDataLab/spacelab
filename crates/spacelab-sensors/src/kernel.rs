//! Batch kernels: one satellite, many cells.
//!
//! These are the functions the engine spends its life in. `g` runs a few dozen times per event;
//! these run once per `(satellite, time, candidate cell)` — of order `10^11` times for a
//! catalog-scale query.
//!
//! Every kernel here:
//!
//! * takes the grid as six contiguous `f64` streams and writes into a caller-provided buffer, so
//!   nothing allocates and nothing is copied;
//! * hoists everything that depends only on the satellite out of the cell loop;
//! * contains no square root, no division, no trigonometry and no data-dependent branch, so the
//!   autovectoriser can lift it to whatever vector width the target has.
//!
//! There are no intrinsics here on purpose. Hand-written AVX-512 would be a code path that this
//! machine cannot execute — the host is Zen 3 (AVX2, no AVX-512) and QEMU's TCG does not
//! implement AVX-512 either, so it could never be run, let alone tested. One source, compiled per
//! target feature, is one thing to test. See `plans/08-parallelism.md`.

use spacelab_core::Vec3;

use crate::{Boresight, Cone, ElevationMask, SensorError, cone, elevation, grid::CellGrid};

/// Check an output buffer against the grid it will be written through.
fn check_len(cells: &CellGrid<'_>, buffer: usize) -> Result<usize, SensorError> {
    let n = cells.len();
    if n == buffer {
        Ok(n)
    } else {
        Err(SensorError::OutputLengthMismatch { cells: n, buffer })
    }
}

/// Write `1` where the satellite clears the elevation mask, `0` elsewhere.
///
/// # Errors
///
/// [`SensorError::OutputLengthMismatch`] if `out` is not exactly one byte per cell.
#[allow(clippy::needless_range_loop)]
pub fn elevation_covered_into(
    mask: ElevationMask,
    satellite: Vec3,
    cells: &CellGrid<'_>,
    out: &mut [u8],
) -> Result<(), SensorError> {
    let n = check_len(cells, out.len())?;
    let (px, py, pz, zx, zy, zz) = cells.columns();
    let (px, py, pz) = (&px[..n], &py[..n], &pz[..n]);
    let (zx, zy, zz) = (&zx[..n], &zy[..n], &zz[..n]);
    let out = &mut out[..n];

    let s = mask.sin_min();
    let s2 = s * s;
    let (rx, ry, rz) = (satellite.x, satellite.y, satellite.z);

    // The branch on the sign of the mask is loop-invariant, so it is written once, outside.
    // Below the horizon the inequality flips; see `ElevationMask`.
    if s >= 0.0 {
        for i in 0..n {
            let (dx, dy, dz) = (rx - px[i], ry - py[i], rz - pz[i]);
            let t = dx * zx[i] + dy * zy[i] + dz * zz[i];
            let q = dx * dx + dy * dy + dz * dz;
            out[i] = u8::from(t >= 0.0 && t * t >= s2 * q);
        }
    } else {
        for i in 0..n {
            let (dx, dy, dz) = (rx - px[i], ry - py[i], rz - pz[i]);
            let t = dx * zx[i] + dy * zy[i] + dz * zz[i];
            let q = dx * dx + dy * dy + dz * dz;
            out[i] = u8::from(t >= 0.0 || t * t <= s2 * q);
        }
    }
    Ok(())
}

/// Increment `counters[i]` for every cell the satellite currently sees.
///
/// This is coverage multiplicity: fold it over the epoch grid and the counter *is* the metric.
/// Accumulating in place avoids materialising a per-epoch coverage mask, which for `10^6` cells
/// and `10^5` epochs would be 100 GB nobody wants.
///
/// # Errors
///
/// [`SensorError::OutputLengthMismatch`] if `counters` is not exactly one per cell.
#[allow(clippy::needless_range_loop)]
pub fn elevation_accumulate_into(
    mask: ElevationMask,
    satellite: Vec3,
    cells: &CellGrid<'_>,
    counters: &mut [u32],
) -> Result<(), SensorError> {
    let n = check_len(cells, counters.len())?;
    let (px, py, pz, zx, zy, zz) = cells.columns();
    let (px, py, pz) = (&px[..n], &py[..n], &pz[..n]);
    let (zx, zy, zz) = (&zx[..n], &zy[..n], &zz[..n]);
    let counters = &mut counters[..n];

    let s = mask.sin_min();
    let s2 = s * s;
    let (rx, ry, rz) = (satellite.x, satellite.y, satellite.z);
    let below = s < 0.0;

    for i in 0..n {
        let (dx, dy, dz) = (rx - px[i], ry - py[i], rz - pz[i]);
        let t = dx * zx[i] + dy * zy[i] + dz * zz[i];
        let q = dx * dx + dy * dy + dz * dz;
        let up = t >= 0.0;
        let sq = t * t >= s2 * q;
        let hit = if below { up || !sq } else { up && sq };
        counters[i] += u32::from(hit);
    }
    Ok(())
}

/// Write `1` where the cell lies inside the nadir cone, `0` elsewhere.
///
/// This is the cone alone. A cell behind the limb can satisfy it; compose with
/// [`crate::And`] and [`ElevationMask::HORIZON`] for a physically visible footprint.
///
/// # Errors
///
/// [`SensorError::OutputLengthMismatch`] if `out` is not exactly one byte per cell.
#[allow(clippy::needless_range_loop)]
pub fn cone_covered_into(
    sensor: Cone,
    satellite: Vec3,
    cells: &CellGrid<'_>,
    out: &mut [u8],
) -> Result<(), SensorError> {
    let n = check_len(cells, out.len())?;
    let (px, py, pz, _, _, _) = cells.columns();
    let (px, py, pz) = (&px[..n], &py[..n], &pz[..n]);
    let out = &mut out[..n];

    let (rx, ry, rz) = (satellite.x, satellite.y, satellite.z);
    let c2 = sensor.cos_half() * sensor.cos_half();

    match sensor.boresight() {
        Boresight::GeocentricNadir => {
            let n2 = rx * rx + ry * ry + rz * rz; // hoisted: per satellite, not per cell
            for i in 0..n {
                let (cx, cy, cz) = (px[i], py[i], pz[i]);
                let utilde = n2 - (cx * rx + cy * ry + cz * rz);
                let (dx, dy, dz) = (rx - cx, ry - cy, rz - cz);
                let q = dx * dx + dy * dy + dz * dz;
                out[i] = u8::from(utilde >= 0.0 && utilde * utilde >= c2 * q * n2);
            }
        }
        // The branch is on the sensor, so it is hoisted out of the loop by construction rather than
        // by hoping the optimiser lifts it. The pointed arm is one multiply *cheaper* per cell: a
        // supplied boresight is already unit, so the `n2` that existed only to avoid normalising `r`
        // is not needed.
        Boresight::Fixed(b) => {
            for i in 0..n {
                let (dx, dy, dz) = (px[i] - rx, py[i] - ry, pz[i] - rz);
                let u = dx * b.x + dy * b.y + dz * b.z;
                let q = dx * dx + dy * dy + dz * dz;
                out[i] = u8::from(u >= 0.0 && u * u >= c2 * q);
            }
        }
    }
    Ok(())
}

/// Scalar reference: elevation coverage for one cell, through `elevation::covered_from`.
///
/// Exists so tests can assert the batch kernel and the per-cell predicate agree.
#[must_use]
pub fn elevation_covered_one(
    mask: ElevationMask,
    satellite: Vec3,
    cell: Vec3,
    zenith: Vec3,
) -> bool {
    let geom = crate::AccessGeometry {
        satellite,
        cell,
        zenith,
    };
    let (t, q) = elevation::dots(&geom);
    elevation::covered_from(t, q, mask.sin_min())
}

/// Scalar reference: cone coverage for one cell, through `cone::covered_from`.
#[must_use]
pub fn cone_covered_one(sensor: Cone, satellite: Vec3, cell: Vec3) -> bool {
    let geom = crate::AccessGeometry {
        satellite,
        cell,
        zenith: Vec3::ZERO, // unused by the cone
    };
    match sensor.boresight() {
        Boresight::GeocentricNadir => {
            let (utilde, q, n2) = cone::terms(&geom);
            cone::covered_from(utilde, q, n2, sensor.cos_half())
        }
        Boresight::Fixed(b) => {
            let (u, q) = cone::terms_pointed(&geom, b);
            cone::covered_from_pointed(u, q, sensor.cos_half())
        }
    }
}
