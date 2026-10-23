//! Criterion benchmarks for the two costs that dominate a numerical orbit propagation: the per-stage
//! geopotential force evaluation, and a full adaptive integration over one orbit.
//!
//! `cargo bench -p spacelab-propagation`
//!
//! * `geopotential_acceleration` — the tesseral force (XGM2019e), scalar vs `f64x4` SIMD, swept by
//!   degree. This is the single call every integrator stage makes; the SIMD/scalar ratio and the
//!   degree trade-off are what a fidelity-vs-cost decision turns on.
//! * `dop853_two_body_one_orbit` — a complete Dormand–Prince 8(5,3) adaptive integration of the planar
//!   two-body problem over one period, with dense output — the whole-propagation cost, not one step.

#![allow(clippy::expect_used, clippy::doc_markdown)] // bench-only expect; method names read fine as prose

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::Acceleration;
use spacelab_propagation::{Dop853, DynamicalSystem, GeoPotential, Tolerance};
use std::hint::black_box;

fn bench_geopotential(c: &mut Criterion) {
    let r = [6_778_000.0_f64, 1_200_000.0, 900_000.0];
    let v = [0.0_f64; 3];
    let mut group = c.benchmark_group("geopotential_acceleration");
    for &degree in &[8usize, 20, 70, 120] {
        let g = GeoPotential::embedded(degree).expect("degree within the embedded model");
        group.bench_with_input(BenchmarkId::new("scalar", degree), &degree, |b, _| {
            b.iter(|| black_box(g.acceleration(0.0, black_box(&r), &v)));
        });
        group.bench_with_input(BenchmarkId::new("simd", degree), &degree, |b, _| {
            b.iter(|| black_box(g.acceleration_simd(black_box(&r))));
        });
    }
    group.finish();
}

/// Planar two-body dynamics `(x, y, vx, vy)` in km, km/s (Earth `mu` in km³/s²).
struct TwoBody {
    mu: f64,
}

impl DynamicalSystem<4> for TwoBody {
    fn derivative(&self, _t: f64, y: &[f64; 4], dydt: &mut [f64; 4]) {
        let r = (y[0] * y[0] + y[1] * y[1]).sqrt();
        let r3 = r * r * r;
        dydt[0] = y[2];
        dydt[1] = y[3];
        dydt[2] = -self.mu * y[0] / r3;
        dydt[3] = -self.mu * y[1] / r3;
    }
}

fn bench_propagator(c: &mut Criterion) {
    let mu = 398_600.441_8_f64; // km³/s²
    let r = 7000.0_f64;
    let v = (mu / r).sqrt();
    let period = core::f64::consts::TAU * (r * r * r / mu).sqrt();
    let sys = TwoBody { mu };
    let y0 = [r, 0.0, 0.0, v];
    c.bench_function("dop853_two_body_one_orbit", |b| {
        b.iter(|| {
            black_box(
                integrate(
                    &Dop853,
                    &sys,
                    0.0,
                    &y0,
                    period,
                    period / 100.0,
                    Controls::new(Tolerance::new(1e-12, 1e-12), 1e-9),
                )
                .expect("two-body integration converges"),
            )
        });
    });
}

criterion_group!(benches, bench_geopotential, bench_propagator);
criterion_main!(benches);
