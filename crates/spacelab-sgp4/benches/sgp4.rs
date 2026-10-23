//! Criterion benchmarks for SGP4: TLE parse + element setup, and steady-state propagation.
//!
//! `cargo bench -p spacelab-sgp4`
//!
//! The TLE is the canonical Vallado ISS case from the official SGP4 verification set. `from_tle` pays
//! the one-time setup (Brouwer mean-motion recovery, secular/periodic coefficient tables); a
//! `propagate_minutes` call is the per-timestep cost a catalog-scale query multiplies by N×T.

#![allow(clippy::unwrap_used)] // the fixed verification TLE parses and propagates by construction

use criterion::{Criterion, criterion_group, criterion_main};
use spacelab_sgp4::Satellite;
use std::hint::black_box;

const LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
const LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

fn bench_sgp4(c: &mut Criterion) {
    c.bench_function("sgp4/from_tle_setup", |b| {
        b.iter(|| black_box(Satellite::from_tle(black_box(LINE1), black_box(LINE2)).unwrap()));
    });

    let sat = Satellite::from_tle(LINE1, LINE2).unwrap();
    c.bench_function("sgp4/propagate_minutes", |b| {
        let mut minutes = 0.0_f64;
        b.iter(|| {
            minutes += 1.0; // sweep along the orbit so each call is a distinct time
            black_box(sat.propagate_minutes(black_box(minutes)).unwrap())
        });
    });
}

criterion_group!(benches, bench_sgp4);
criterion_main!(benches);
