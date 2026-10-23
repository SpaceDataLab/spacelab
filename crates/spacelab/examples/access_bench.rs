//! Throughput of the access-window engine (`revisit_map`), its thread scaling, and the spatial cull.
//!
//! `cargo run -p spacelab --release --example access_bench`
//!
//! The catalog-scale MVP query — for each ground cell, detect every access window of a satellite over
//! a horizon by bracketing sign changes of the constraint `g(t)` on a derived coarse grid and refining
//! with Brent. This measures (1) cell-seconds per wall second and the 1→N-thread speed-up, and (2) the
//! spatial cull: `reachable_band` skips cells a circular orbit can never see, so the event search runs
//! only on candidates. The cull is asserted to never drop a covered cell.

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::trivially_copy_pass_by_ref
)]

use std::hint::black_box;
use std::time::Instant;

use spacelab_core::{Mat3, Vec3};
use spacelab_coverage::{CoverageMetrics, revisit_map};
use spacelab_frames::EARTH_ROTATION_RATE;
use spacelab_sensors::{CullBody, ElevationMask, OrbitBound, reachable_band};

const RE: f64 = 6_378_137.0;
const MU: f64 = 3.986_004_418e14;

/// A near-uniform sphere of `n` ground cells (Fibonacci lattice), each `(centre, zenith)`.
fn cells(n: usize) -> Vec<(Vec3, Vec3)> {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let th = golden * i as f64;
            let up = Vec3::new(r * th.cos(), r * th.sin(), z);
            (up.scale(RE), up)
        })
        .collect()
}

fn sat_closure(a: f64, inc: f64) -> impl Fn(f64) -> Vec3 + Sync + Copy {
    let n = (MU / (a * a * a)).sqrt();
    move |t: f64| {
        let u = n * t;
        let (cu, su) = (u.cos(), u.sin());
        let (ci, si) = (inc.cos(), inc.sin());
        Mat3::rot3(EARTH_ROTATION_RATE * t) * Vec3::new(a * cu, a * su * ci, a * su * si)
    }
}

fn run<P: Fn(f64) -> Vec3 + Sync>(
    grid: &[(Vec3, Vec3)],
    sensor: &ElevationMask,
    orbit: &OrbitBound,
    sat: P,
    horizon: f64,
    threads: usize,
) -> (f64, Vec<CoverageMetrics>) {
    let t0 = Instant::now();
    let m = revisit_map(
        grid,
        sensor,
        orbit,
        EARTH_ROTATION_RATE,
        sat,
        0.0,
        horizon,
        1.0,
        threads,
    );
    (t0.elapsed().as_secs_f64(), m)
}

fn main() {
    let n = 4096;
    let horizon = 86_400.0;
    let grid = cells(n);
    let a = RE + 550e3;
    let sensor = ElevationMask::new(5.0_f64.to_radians()).expect("mask in range");
    let orbit = OrbitBound {
        semi_major_axis: a,
        eccentricity: 0.0,
        mu: MU,
    };

    println!(
        "access-window engine (revisit_map): {n} cells × {:.0} h, ElevationMask(5°)\n",
        horizon / 3600.0
    );

    println!("thread scaling (inc = 53°):");
    let sat53 = sat_closure(a, 53.0_f64.to_radians());
    let mut single = 0.0;
    for threads in [1usize, 2, 4, 8] {
        let (dt, m) = run(&grid, &sensor, &orbit, sat53, horizon, threads);
        black_box(&m);
        if threads == 1 {
            single = dt;
        }
        println!(
            "  {threads} thread(s): {:>7.1} ms   {:>6.2}e6 cell·s/s   speedup {:>4.2}×",
            dt * 1e3,
            n as f64 * horizon / dt / 1e6,
            single / dt,
        );
    }

    println!("\nspatial cull (reachable_band, 8 threads):");
    for inc_deg in [25.0_f64, 53.0, 98.0] {
        let inc = inc_deg.to_radians();
        let sat = sat_closure(a, inc);
        let (dt_full, full) = run(&grid, &sensor, &orbit, sat, horizon, 8);

        let band = reachable_band(inc, a, &sensor, CullBody::WGS84);
        let sin_band = band.sin();
        let keep: Vec<usize> = (0..n).filter(|&i| grid[i].1.z.abs() <= sin_band).collect();
        // correctness: a culled cell must never have been covered in the full run.
        let culled_covered = (0..n)
            .filter(|&i| grid[i].1.z.abs() > sin_band)
            .filter(|&i| full[i].passes > 0)
            .count();
        assert_eq!(
            culled_covered, 0,
            "cull dropped {culled_covered} covered cells at inc {inc_deg}°"
        );

        let kept_grid: Vec<(Vec3, Vec3)> = keep.iter().map(|&i| grid[i]).collect();
        let (dt_cull, _) = run(&kept_grid, &sensor, &orbit, sat, horizon, 8);
        let culled_pct = 100.0 * (n - keep.len()) as f64 / n as f64;
        println!(
            "  inc {inc_deg:>4.0}°: band ±{:>4.1}°  culled {culled_pct:>4.1}%   full {:>6.1} ms → culled {:>6.1} ms   {:>4.2}× ",
            band.to_degrees(),
            dt_full * 1e3,
            dt_cull * 1e3,
            dt_full / dt_cull,
        );
    }
}
