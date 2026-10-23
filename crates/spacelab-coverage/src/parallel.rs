//! Parallel per-cell revisit, partitioning the area of interest across threads.
//!
//! Unlike the batch coverage kernel — memory-bandwidth-bound, so it barely scales past a few threads
//! (`plans/08-parallelism.md`) — the per-cell **access-window** work is compute-bound: each cell runs
//! its own adaptive scan and root-finds, independent of every other. That is embarrassingly parallel,
//! and partitioning the cells across threads scales with cores. The reduction is exact: the result is
//! bit-for-bit what a serial pass produces, because the per-cell computations do not interact.
//!
//! Scoped threads (`std::thread::scope`) borrow the trajectory and sensor directly, so nothing is
//! cloned or reference-counted; this is the reason the module is `std`-only.

// The driver's arguments are each a distinct quantity — the cells, the sensor, the orbit bound, the
// frame rotation, the trajectory, the scan window, the thread count. A struct would hide the seam.
#![allow(clippy::too_many_arguments)]

use spacelab_core::Vec3;
use spacelab_sensors::{Constraint, OrbitBound};

use crate::access::access_windows;
use crate::metrics::{CoverageMetrics, metrics};

/// The revisit metrics of every cell, computed across `threads` scoped threads.
///
/// `cells` is one `(centre, zenith)` per ground cell in the frame `frame_rate` rotates. `sat_at`
/// gives the satellite position in that frame over time (both `Sync`, shared across threads).
/// Returns one [`CoverageMetrics`] per cell, in the input order — identical to running each cell
/// serially.
#[must_use]
pub fn revisit_map<C, P>(
    cells: &[(Vec3, Vec3)],
    sensor: &C,
    orbit: &OrbitBound,
    frame_rate: f64,
    sat_at: P,
    t0: f64,
    t1: f64,
    tol: f64,
    threads: usize,
) -> Vec<CoverageMetrics>
where
    C: Constraint + Sync,
    P: Fn(f64) -> Vec3 + Sync,
{
    let n = cells.len();
    let mut out = vec![metrics(&[], t0, t1); n];
    if n == 0 {
        return out;
    }
    let chunk = n.div_ceil(threads.max(1));
    let sat_at = &sat_at;

    std::thread::scope(|scope| {
        for (out_chunk, cell_chunk) in out.chunks_mut(chunk).zip(cells.chunks(chunk)) {
            scope.spawn(move || {
                for (slot, &(center, zenith)) in out_chunk.iter_mut().zip(cell_chunk) {
                    let mut windows = Vec::new();
                    access_windows(
                        sensor,
                        orbit,
                        frame_rate,
                        center,
                        zenith,
                        sat_at,
                        t0,
                        t1,
                        tol,
                        |w| {
                            windows.push(w);
                        },
                    );
                    *slot = metrics(&windows, t0, t1);
                }
            });
        }
    });

    out
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::many_single_char_names,
        clippy::cast_precision_loss
    )]

    use super::revisit_map;
    use spacelab_core::Vec3;
    use spacelab_sensors::{ElevationMask, OrbitBound};

    const RE: f64 = 6_378_137.0;
    const MU: f64 = 3.986_004_418e14;

    /// Ground cells spread around the equator, each `(centre, zenith)` (radial on a sphere).
    fn ring_of_cells(count: usize) -> Vec<(Vec3, Vec3)> {
        (0..count)
            .map(|i| {
                let theta = core::f64::consts::TAU * (i as f64) / (count as f64);
                let up = Vec3::new(theta.cos(), theta.sin(), 0.0);
                (up.scale(RE), up)
            })
            .collect()
    }

    #[test]
    fn the_parallel_map_equals_the_serial_map_cell_for_cell() {
        let cells = ring_of_cells(24);
        let r = 7_000e3;
        let omega = (MU / (r * r * r)).sqrt();
        let sat_at = |t: f64| Vec3::new(r * (omega * t).cos(), r * (omega * t).sin(), 0.0);
        let mask = ElevationMask::new(5.0_f64.to_radians()).unwrap();
        let orbit = OrbitBound {
            semi_major_axis: r,
            eccentricity: 0.0,
            mu: MU,
        };
        let span = 3.0 * core::f64::consts::TAU / omega;

        let serial = revisit_map(&cells, &mask, &orbit, 0.0, sat_at, 0.0, span, 1e-3, 1);
        let parallel = revisit_map(&cells, &mask, &orbit, 0.0, sat_at, 0.0, span, 1e-3, 8);

        // deterministic and non-interacting, so the partition changes nothing — bit for bit
        assert_eq!(
            serial, parallel,
            "the threaded partition changed the result"
        );
        // One entry per cell, which is also what makes the sweep below non-vacuous: `all` over an
        // empty map is true, so a function that returned nothing would satisfy it.
        assert_eq!(serial.len(), cells.len(), "one metric per cell, in order");
        assert!(
            serial.iter().all(|m| m.passes >= 1),
            "some equatorial cell was never seen"
        );
    }

    #[test]
    fn an_empty_area_of_interest_yields_an_empty_map() {
        let sat_at = |_t: f64| Vec3::new(7_000e3, 0.0, 0.0);
        let orbit = OrbitBound {
            semi_major_axis: 7_000e3,
            eccentricity: 0.0,
            mu: MU,
        };
        let map = revisit_map(
            &[],
            &ElevationMask::HORIZON,
            &orbit,
            0.0,
            sat_at,
            0.0,
            1000.0,
            1e-3,
            4,
        );
        assert_eq!(map.len(), 0);
    }
}
