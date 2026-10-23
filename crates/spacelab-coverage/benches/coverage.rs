//! Criterion benchmark for the access-window engine (`revisit_map`) — the catalog-scale coverage query.
//!
//! `cargo bench -p spacelab-coverage`
//!
//! For each ground cell the engine brackets sign changes of the elevation constraint over the horizon
//! and refines each access with Brent. Cost scales with cells × horizon; this sweeps the cell count on
//! a single thread (deterministic timing) at a fixed 24 h horizon. The satellite is a closed-form
//! circular-orbit closure, so the measurement isolates the event search, not the propagator.

#![allow(clippy::cast_precision_loss, clippy::expect_used)] // bench-only expect on an in-range mask

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use spacelab_core::{Mat3, Vec3};
use spacelab_coverage::revisit_map;
use spacelab_sensors::{ElevationMask, OrbitBound};
use std::hint::black_box;

const RE: f64 = 6_378_137.0;
const MU: f64 = 3.986_004_418e14;
// IERS mean Earth rotation rate (rad/s); hardcoded to keep the bench free of a frames dev-dep.
const EARTH_ROTATION_RATE: f64 = 7.292_115_146_706_979e-5;

/// A near-uniform Fibonacci-lattice sphere of `n` ground cells, each `(surface point, zenith)`.
fn cells(n: usize) -> Vec<(Vec3, Vec3)> {
    let golden = core::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
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

/// A circular orbit at semi-major axis `a`, inclination `inc`, under a spinning Earth.
fn sat_closure(a: f64, inc: f64) -> impl Fn(f64) -> Vec3 + Sync + Copy {
    let n = (MU / (a * a * a)).sqrt();
    move |t: f64| {
        let u = n * t;
        let (cu, su) = (u.cos(), u.sin());
        let (ci, si) = (inc.cos(), inc.sin());
        Mat3::rot3(EARTH_ROTATION_RATE * t) * Vec3::new(a * cu, a * su * ci, a * su * si)
    }
}

fn bench_revisit_map(c: &mut Criterion) {
    let a = RE + 550e3;
    let horizon = 86_400.0;
    let sensor = ElevationMask::new(5.0_f64.to_radians()).expect("mask in range");
    let orbit = OrbitBound {
        semi_major_axis: a,
        eccentricity: 0.0,
        mu: MU,
    };
    let sat = sat_closure(a, 53.0_f64.to_radians());

    let mut group = c.benchmark_group("revisit_map");
    group.sample_size(20); // each iteration is a full 24 h × N-cell sweep — keep wall time bounded
    for &n in &[256usize, 1024] {
        let grid = cells(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                black_box(revisit_map(
                    &grid,
                    &sensor,
                    &orbit,
                    EARTH_ROTATION_RATE,
                    sat,
                    0.0,
                    horizon,
                    1.0,
                    1,
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_revisit_map);
criterion_main!(benches);
